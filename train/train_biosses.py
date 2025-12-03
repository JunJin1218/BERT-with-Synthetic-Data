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
    TrainingArguments,
    Trainer,
)
import numpy as np
from evaluate import load as load_metric
import torch

from utils.utils import preprocess_function


def get_task_name(cfg):
    return cfg.task

def data_preprocess(cfg: DictConfig):
    """
    BIOSSES 전용 데이터 준비:
      - original : 원본 train/test 사용
      - synthetic: 합성 train을 scale 만큼 샘플링
      - mix      : 원본 train + 합성 샘플 결합
    점수(score)는 [0,4] 연속값 회귀 라벨(float)로 사용.
    """
    # 1) Load original dataset
    original_train_path = os.path.join("data", "biosses", "original", "train.jsonl")
    original_test_path  = os.path.join("data", "biosses", "original", "test.jsonl")

    original_train = load_dataset("json", data_files={"train": original_train_path}, split="train")
    original_test  = load_dataset("json", data_files={"test":  original_test_path},  split="test")

    original_size = len(original_train)
    data_scale = int(cfg.scale * original_size)   # scale can be >= 1

    # 2) Load synthetic dataset (optional for modes)
    synthetic_path = os.path.join("data", get_task_name(cfg), cfg.data_model, cfg.dataset)
    if not os.path.exists(synthetic_path) and cfg.mode in ["synthetic", "mix"]:
        raise FileNotFoundError(f"Synthetic data not found: {synthetic_path}")

    if os.path.exists(synthetic_path):
        synthetic_ds = load_dataset("json", data_files={"train": synthetic_path}, split="train")
        # 회귀 라벨: score 컬럼이 존재한다고 가정 (float로 캐스팅)
        if "score" not in synthetic_ds.column_names:
            raise KeyError("Synthetic dataset must include a 'score' field for regression.")
        synthetic_ds = synthetic_ds.map(lambda ex: {"score": float(ex["score"])})  # 안전장치
        synthetic_ds = synthetic_ds.cast_column("score", Value("float64"))

    else:
        synthetic_ds = None

    # --- Mode selection ---
    mode = cfg.mode  # "original" / "synthetic" / "mix"
    if mode == "original":
        final_train = original_train

    elif mode == "synthetic":
        if len(synthetic_ds) < data_scale:
            raise ValueError(
                f"Requested {data_scale} synthetic samples but only {len(synthetic_ds)} available."
            )
        final_train = synthetic_ds.shuffle(seed=cfg.seed).select(range(data_scale))

    elif mode == "mix":
        if len(synthetic_ds) < data_scale:
            raise ValueError(
                f"Requested {data_scale} synthetic samples but only {len(synthetic_ds)} available."
            )
        sampled_syn = synthetic_ds.shuffle(seed=cfg.seed).select(range(data_scale))
        final_train = concatenate_datasets([original_train, sampled_syn]).shuffle(seed=cfg.seed)
    else:
        raise ValueError(f"Invalid cfg.mode: {mode}. Expected 'original', 'synthetic', or 'mix'.")

    # 최종
    return DatasetDict({"train": final_train, "test": original_test})


@hydra.main(version_base=None, config_path="configs", config_name="setting")
def main(cfg: DictConfig):
    assert cfg.task == "biosses", "This script is now BIOSSES-only."
    task_name = get_task_name(cfg)

    # 1. 토크나이저 / 데이터
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    data = data_preprocess(cfg)

    # 2. 전처리: preprocess_function이 sentence1/sentence2/score -> 'labels'(float)로 매핑하도록 구현되어 있어야 함
    encoded_datasets = data.map(
        lambda examples: preprocess_function(cfg, examples, tokenizer),
        batched=True,
        remove_columns=[c for c in data["train"].column_names if c not in ("labels", "input_ids", "token_type_ids", "attention_mask")],
    )

    # 3. 모델 (회귀)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=1)
    model.config.problem_type = "regression"

    # 4. Metrics (Pearson/Spearman + MSE)
    pearson = load_metric("pearsonr")
    spearman = load_metric("spearmanr")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = np.array(labels, dtype=np.float32)
        preds = np.array(logits).reshape(-1)  # (N,1)->(N,)

        # 평가지표 계산 (필요시 범위 클램프는 지표용으로만)
        preds_clamped = np.clip(preds, 0.0, 4.0)
        out = {
            "pearson":  pearson.compute(predictions=preds_clamped, references=labels)["pearsonr"],
            "spearman": spearman.compute(predictions=preds_clamped, references=labels)["spearmanr"],
            "mse":      float(np.mean((preds - labels) ** 2)),
        }
        return out

    # 5. TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"./outputs/{task_name}",
        eval_strategy=cfg.eval_strategy,  # <-- param name 수정
        save_strategy=cfg.save_strategy,
        learning_rate=cfg.learning_rate,
        save_total_limit=cfg.save_total_limit,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,  # 보통 "pearson"
        greater_is_better=cfg.greater_is_better,          # 보통 True
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        report_to="none",
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7. 학습
    trainer.train()

    # 8. 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = (
        f"./models/{task_name}/{cfg.mode}/{timestamp}"
        if cfg.mode == "original"
        else f"./models/{task_name}/{cfg.mode}/{cfg.scale}/{timestamp}"
    )
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # 9. 평가 및 results.json 기록
    val_results = trainer.evaluate(encoded_datasets["test"])
    print("Validation results:", val_results)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    metric_name = cfg.metric_for_best_model
    best_metric = trainer.state.best_metric
    best_checkpoint = trainer.state.best_model_checkpoint
    best_epoch = None
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
        "config": cfg_dict,
        "best_model": {
            "checkpoint": best_checkpoint,
            "metric_name": metric_name,
            "metric_value": best_metric,
            "epoch": best_epoch,
        },
        "validation_results": val_results,
    }

    with open(os.path.join(model_save_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
