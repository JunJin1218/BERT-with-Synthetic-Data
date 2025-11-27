import os
import json
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from evaluate import load as load_metric
import torch


def get_task_name(cfg: DictConfig):
    return f"{cfg.benchmark}-{cfg.task}"


def data_preprocess(cfg: DictConfig):
    """
    - HF hub에서 원본 데이터셋 로드
    - synthetic.jsonl 있으면 train split에만 합침
    - validation / test는 원본 그대로 사용
    - 최종적으로 DatasetDict 형태로 반환
    """
    # 1) 원본 SuperGLUE (예: super_glue, 'cb') 로드
    raw_datasets = load_dataset(cfg.benchmark, cfg.task)
    train_split = raw_datasets["train"]
    validation_split = raw_datasets["validation"]
    test_split = raw_datasets["test"] if "test" in raw_datasets else None

    # 2) synthetic data 경로 구성
    synthetic_data_path = os.path.join(
        "data",
        get_task_name(cfg),  # e.g) "super_glue-cb"
        cfg.data_model,      # e.g) "gpt-4o-mini"
        "synthetic.jsonl",
    )

    # 3) synthetic.jsonl 있으면 train에 concat
    if os.path.exists(synthetic_data_path):
        # jsonl → HF Dataset
        # 각 라인은 {"premise": ..., "hypothesis": ..., "label": ...} 형태라고 가정
        synthetic_ds = load_dataset(
            "json",
            data_files={"train": synthetic_data_path},
            split="train",
        )

        # 1) 원본 label 스키마(ClassLabel) 가져오기
        original_label_feature = train_split.features["label"]

        # 2) synthetic label(int)을 ClassLabel(int)로 캐스팅
        synthetic_ds = synthetic_ds.cast_column("label", original_label_feature)

        # 3) concat
        train_split = concatenate_datasets([train_split, synthetic_ds])
    else:
        print(f"[data_preprocess] synthetic data not found at: {synthetic_data_path}")

    # 4) DatasetDict로 묶어서 반환
    dataset_dict_kwargs = {
        "train": train_split,
        "validation": validation_split,
    }
    if test_split is not None:
        dataset_dict_kwargs["test"] = test_split

    return DatasetDict(dataset_dict_kwargs)


# 전처리 함수: premise + hypothesis → BERT 입력
def preprocess_function(cfg: DictConfig, examples, tokenizer):
    # TODO
    LABEL_DICT = {
        "super_glue-cb": ["premise", "hypothesis"],
    }
    task_name = get_task_name(cfg)
    premise_key, hypothesis_key = LABEL_DICT[task_name]

    return tokenizer(
        examples[premise_key],
        examples[hypothesis_key],
        truncation=cfg.truncation,
        padding=cfg.padding,
        max_length=cfg.max_length,
    )


@hydra.main(version_base=None, config_path=".", config_name="setting")
def main(cfg: DictConfig):
    # 1. 토크나이저 / 데이터 불러오기
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    data = data_preprocess(cfg)

    # CB label 이름 (entailment / contradiction / neutral 등)
    label_list = data["train"].features["label"].names
    num_labels = len(label_list)

    # 2. 전처리 적용
    encoded_datasets = data.map(
        lambda examples: preprocess_function(cfg, examples, tokenizer),
        batched=True,
    )

    # 3. 모델 로드: BERT + SequenceClassification Head
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model,
        num_labels=num_labels,
    )

    # 4. Metric 정의 (accuracy + macro F1)
    accuracy = load_metric("accuracy")
    f1 = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        f1_score = f1.compute(
            predictions=preds,
            references=labels,
            average="macro",
        )
        acc = accuracy.compute(predictions=preds, references=labels)
        return {
            "accuracy": acc["accuracy"],
            "f1_macro": f1_score["f1"],
        }

    # 5. TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir="./outputs/bert-superglue-cb",
        eval_strategy="epoch",     # 매 epoch마다 eval
        save_strategy="epoch",
        learning_rate=2e-5,
        save_total_limit=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,             # CB는 데이터 작아서 epoch 좀 늘려도 됨
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        gradient_accumulation_steps=2
    )

    # 6. Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7. 학습 시작
    trainer.train()

    # 8. 최종 모델 / 토크나이저 저장
    trainer.save_model("./models/bert-superglue-cb")
    tokenizer.save_pretrained("./models/bert-superglue-cb")

    # 9. Validation 평가
    val_results = trainer.evaluate(encoded_datasets["validation"])
    print("Validation results:", val_results)

    # 10. 예측 예시용 헬퍼
    def predict_example(premise: str, hypothesis: str) -> str:
        model.eval()
        device = model.device  # cuda:0 또는 cpu

        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        # 입력 텐서를 모델이 있는 디바이스로 옮기기
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = logits.argmax(dim=-1).item()
        return label_list[pred_id]

    # 사용 예시 (원하면 주석 해제해서 quick test)
    print("="*20)
    print("Prediction Test")
    print(predict_example("It is raining today.", "The weather is sunny."))


if __name__ == "__main__":
    main()
