import os
import json
from tqdm import tqdm
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from datasets import load_dataset, DatasetDict, concatenate_datasets, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
)
import numpy as np
from evaluate import load as load_metric
import torch

from utils.utils import get_task_name, preprocess_function


def data_preprocess(cfg: DictConfig):
    """
    Prepare dataset according to mode:
      - original : use only original dataset
      - synthetic: use only synthetic dataset (sampled by scale)
      - mix      : sampled synthetic + full original dataset
    scale >= 1 is allowed (scale * original_train_size).
    """

    # 1) Load original dataset
    raw_datasets = load_dataset(cfg.benchmark, cfg.task)
    for split, ds in raw_datasets.items():
        for col in ("span1_index", "span2_index"):
            if col in ds.column_names:
                raw_datasets[split] = ds.cast_column(col, Value("int32"))
    original_train = raw_datasets["train"]
    validation_split = raw_datasets["validation"]
    test_split = raw_datasets["test"] if "test" in raw_datasets else None

    original_size = len(original_train)
    data_scale = int(cfg.scale * original_size)   # scale can be >= 1

    # 2) Load synthetic dataset
    synthetic_path = os.path.join(
        "data",
        get_task_name(cfg),
        cfg.data_model,
        cfg.dataset,
    )

    if not os.path.exists(synthetic_path):
        raise FileNotFoundError(f"Synthetic data not found: {synthetic_path}")

    synthetic_ds = load_dataset(
        "json",
        data_files={"train": synthetic_path},
        split="train",
    )
    for col in ("span1_index", "span2_index"):
        if col in synthetic_ds.column_names:
            synthetic_ds = synthetic_ds.cast_column(col, Value("int32"))

    # Cast synthetic label to ClassLabel schema
    original_label_feature = original_train.features["label"]
    synthetic_ds = synthetic_ds.cast_column("label", original_label_feature)

    # --- Mode selection -------------------------------------------------

    mode = cfg.mode  # "original" / "synthetic" / "mix"

    if mode == "original":
        # Use only original dataset
        final_train = original_train

    elif mode == "synthetic":
        # Use only synthetic data (sample by data_scale)
        if len(synthetic_ds) < data_scale:
            raise ValueError(
                f"Requested {data_scale} synthetic samples but only {len(synthetic_ds)} available."
            )
        final_train = synthetic_ds.shuffle(seed=cfg.seed).select(range(data_scale))

    elif mode == "mix":
        # Sample synthetic by scale, then add original dataset
        if len(synthetic_ds) < data_scale:
            raise ValueError(
                f"Requested {data_scale} synthetic samples but only {len(synthetic_ds)} available."
            )

        sampled_syn = synthetic_ds.shuffle(seed=cfg.seed).select(range(data_scale))
        final_train = concatenate_datasets([original_train, sampled_syn])
        final_train = final_train.shuffle(seed=cfg.seed)

    else:
        raise ValueError(f"Invalid cfg.mode: {mode}. Expected 'original', 'synthetic', or 'mix'.")

    # --------------------------------------------------------------------

    return DatasetDict({
        "train": final_train,
        "validation": validation_split,
        "test": test_split
    })


@hydra.main(version_base=None, config_path="configs", config_name="setting")
def main(cfg: DictConfig):
    task_name = get_task_name(cfg)

    # 1. 토크나이저 / 데이터 불러오기
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    data = data_preprocess(cfg)

    # CB label 이름 (entailment / contradiction / neutral 등)
    label_list = data["train"].features["label"].names
    num_labels = len(label_list)

    # 2. 전처리 적용 TODO
    encoded_datasets = data.map(
        lambda examples: preprocess_function(cfg, examples, tokenizer),
        batched=True,
    )

    # 3. 모델 로드: BERT + SequenceClassification Head
    model = ...
    if (cfg.task == "cb"):
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model,
            num_labels=num_labels,
        )
    elif (cfg.task in ["wsc", "copa"]):
        model = AutoModelForMultipleChoice.from_pretrained(
            cfg.model,
            num_labels=num_labels
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
        output_dir=f"./outputs/{task_name}",
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        learning_rate=cfg.learning_rate,
        save_total_limit=cfg.save_total_limit,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.mode == "original":
        model_save_path = f"./models/{task_name}/{cfg.mode}/{timestamp}"
    else:
        model_save_path = f"./models/{task_name}/{cfg.mode}/{cfg.scale}/{timestamp}"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

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

    os.makedirs(model_save_path, exist_ok=True)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    metric_name = cfg.metric_for_best_model
    best_metric = trainer.state.best_metric
    best_checkpoint = trainer.state.best_model_checkpoint
    best_epoch = None

    # log_history에서 best metric과 일치하는 epoch 찾기
    for log in trainer.state.log_history:
        if metric_name in log and log.get(metric_name) == best_metric and "epoch" in log:
            best_epoch = log["epoch"]
            break

    results_payload = {
        "timestamp": timestamp,
        "task_name": task_name,
        "mode": cfg.mode,
        "scale": cfg.scale,
        "data_model": cfg.data_model,
        "config": cfg_dict,  # 현재 Hydra 설정 전체
        "best_model": {
            "checkpoint": best_checkpoint,
            "metric_name": metric_name,
            "metric_value": best_metric,
            "epoch": best_epoch,
        },
        "validation_results": val_results,
    }

    results_path = os.path.join(model_save_path, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
