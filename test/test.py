# python >=3.9
# pip install datasets transformers torch tqdm

import os, json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REQUIRED_MODES = ["original", "mix-1", "mix-2", "synthetic-1", "synthetic-3"]
REQUIRED_TASKS = [
    "AX-b","AX-g","BoolQ","CB","COPA","MultiRC","ReCoRD","RTE","WiC","WSC"
]

def load_model_path(json_path: str) -> dict:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(
            f"[model_path] file not found: {p}. "
            "Create test/model_path.json with a dict mapping {task: model_dir}."
        )
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("[model_path] JSON must be an object/dict of {task: path}.")
    # 간단 검증 + 경고
    for t in REQUIRED_TASKS:
        if t not in data:
            print(f"[model_path][WARN] missing key: {t} (will be skipped if not handled elsewhere)")
    return data

MODEL_PATH = load_model_path("test/model_path.json")

HF_CONFIG = {
    "AX-b": "axb",
    "AX-g": "axg",
    "BoolQ": "boolq",
    "CB": "cb",
    "COPA": "copa",
    "MultiRC": "multirc",
    "ReCoRD": "record",
    "RTE": "rte",
    "WiC": "wic",
    "WSC": "wsc",
}

CLASSIFICATION_TASKS = {"AX-b", "AX-g", "BoolQ", "CB", "COPA", "RTE", "WiC", "WSC"}
BATCH = 32
OUT_DIR = "test/submission"

def ensure_dir(p: str): Path(p).mkdir(parents=True, exist_ok=True)

def load_model(model_dir: str):
    if not os.path.isdir(model_dir): return None, None
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    if torch.cuda.is_available(): model.to("cuda")
    return tok, model

@torch.no_grad()
def batched_logits(model, **enc):
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}
    return model(**enc).logits.detach().cpu()

# ---- label 정규화 ----
def nli_norm(raw: str) -> str:
    r = raw.strip().lower()
    if r in {"entailment", "e"}: return "entailment"
    if r in {"not_entailment", "not-entailment", "contradiction", "neutral", "ne"}:
        # 모델에 따라 not_entailment로만 나가도록 통일
        return "not_entailment"
    return r  # 혹시 모를 케이스

def bool_norm(raw: str | int) -> bool:
    if isinstance(raw, bool): return raw
    r = str(raw).strip().lower()
    return r in {"true", "1", "yes", "y", "positive"}

def run_task(task: str, model_dir: str):
    if task not in HF_CONFIG: 
        print(f"[SKIP] unknown task: {task}"); 
        return
    if task not in CLASSIFICATION_TASKS:
        print(f"[SKIP] {task}: separate pipeline needed (MultiRC/ReCoRD).")
        return
    tok, model = load_model(model_dir)
    if tok is None:
        print(f"[SKIP] no model at {model_dir}")
        return

    ds = load_dataset("super_glue", HF_CONFIG[task])
    test = ds["test"]
    ensure_dir(OUT_DIR)
    out_path = Path(OUT_DIR) / f"{task}.jsonl"

    id2label = getattr(model.config, "id2label", None)

    def write(rows):
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 태스크별 인퍼런스 ----
    open(out_path, "w").close()  # truncate

    if task in {"RTE", "AX-b", "AX-g", "CB"}:
        a = test["premise"]; b = test["hypothesis"]; idxs = test["idx"]
        for i in range(0, len(test), BATCH):
            enc = tok(a[i:i+BATCH], b[i:i+BATCH], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            pred = batched_logits(model, **enc).argmax(-1).tolist()
            rows = []
            for j, p in enumerate(pred):
                raw = id2label[p] if id2label else str(p)
                rows.append({"idx": int(idxs[i+j]), "label": nli_norm(raw)})
            write(rows)

    elif task == "BoolQ":
        a = [f"Question: {q}" for q in test["question"]]
        b = test["passage"]; idxs = test["idx"]
        for i in range(0, len(test), BATCH):
            enc = tok(a[i:i+BATCH], b[i:i+BATCH], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            pred = batched_logits(model, **enc).argmax(-1).tolist()
            rows = []
            for j, p in enumerate(pred):
                raw = id2label[p] if id2label else str(p)
                rows.append({"idx": int(idxs[i+j]), "label": bool_norm(raw)})
            write(rows)

    elif task == "WiC":
        s1 = test["sentence1"]; s2 = test["sentence2"]; idxs = test["idx"]
        for i in range(0, len(test), BATCH):
            enc = tok(s1[i:i+BATCH], s2[i:i+BATCH], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            pred = batched_logits(model, **enc).argmax(-1).tolist()
            rows = []
            for j, p in enumerate(pred):
                raw = id2label[p] if id2label else str(p)
                rows.append({"idx": int(idxs[i+j]), "label": bool_norm(raw)})
            write(rows)

    elif task == "WSC":
        text = test["text"]; idxs = test["idx"]
        for i in range(0, len(test), BATCH):
            enc = tok(text[i:i+BATCH], padding=True, truncation=True,
                      max_length=384, return_tensors="pt")
            pred = batched_logits(model, **enc).argmax(-1).tolist()
            rows = []
            for j, p in enumerate(pred):
                raw = id2label[p] if id2label else str(p)
                rows.append({"idx": int(idxs[i+j]), "label": bool_norm(raw)})
            write(rows)

    elif task == "COPA":
        prem = test["premise"]; c1 = test["choice1"]; c2 = test["choice2"]; idxs = test["idx"]
        def score_batch(p_list, c_list):
            enc = tok(p_list, c_list, padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            logits = batched_logits(model, **enc)
            # 이진 분류 가정: argmax 점수 비교
            return logits.max(-1).values
        for i in range(0, len(test), BATCH):
            s1 = score_batch(prem[i:i+BATCH], c1[i:i+BATCH])
            s2 = score_batch(prem[i:i+BATCH], c2[i:i+BATCH])
            rows = []
            for j, better2 in enumerate((s2 > s1).tolist()):
                rows.append({"idx": int(idxs[i+j]), "label": "2" if better2 else "1"})
            write(rows)

    print(f"[OK] {task} -> {out_path}")

if __name__ == "__main__":
    for task, model_dir in MODEL_PATH.items():
        run_task(task, model_dir)
    print("\n[Done] All JSONL written to:", OUT_DIR)
