# python >= 3.9
# pip install datasets transformers torch

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------

REQUIRED_MODES = [
    "original",
    "mix-1",
    "mix-2",
    "mix-3",
    "synthetic-1",
    "synthetic-3",
]

REQUIRED_TASKS = [
    #"AX-b", "AX-g", "BoolQ", 
    #"CB", 
    # "COPA",
    #"MultiRC", "ReCoRD", "RTE", "WiC", 
    "WSC",
]

# HuggingFace super_glue 서브태스크 이름 매핑
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

# 이 스크립트에서 처리하는 태스크 (MultiRC, ReCoRD는 별도 파이프라인)
CLASSIFICATION_TASKS = {
    "AX-b", "AX-g", "BoolQ", "CB", "COPA", "RTE", "WiC", "WSC"
}

BATCH = 32
OUT_DIR = Path("test/submission")
SUMMARY_FILENAME = "summary_by_accuracy.jsonl"

# ----------------------------------------------------------------------
# 유틸
# ----------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_mode_slug(slug: str) -> Tuple[str, Optional[int]]:
    """
    'original'    -> ('original', None)
    'mix-1'       -> ('mix', 1)
    'synthetic-3' -> ('synthetic', 3)
    """
    if "-" not in slug:
        return slug, None
    base, s = slug.split("-", 1)
    try:
        return base, int(s)
    except ValueError:
        return base, None


def load_summary_rows(task: str) -> List[Dict]:
    """
    models/super_glue-{hfslug}/summary_by_accuracy.jsonl 읽어서 리스트로 반환.
    파일이 없으면 빈 리스트.
    """
    hf_slug = HF_CONFIG.get(task)
    if hf_slug is None:
        return []
    f = Path("models") / f"super_glue-{hf_slug}" / SUMMARY_FILENAME
    if not f.exists():
        print(f"[{task}] no summary file at {f}, skip")
        return []
    rows: List[Dict] = []
    with f.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # 이상한 줄은 그냥 무시
                continue
    return rows


def pick_best_case_path(
    rows: List[Dict],
    want_mode: str,
    want_scale: Optional[int],
) -> Optional[str]:
    """
    summary_by_accuracy.jsonl 내용 중에서
      - mode == want_mode
      - scale == want_scale (None 허용)
    인 것들만 모아서,
      1) rank_by_accuracy 오름차순
      2) accuracy 내림차순
    기준으로 가장 좋은 놈의 case_path를 반환.
    """
    cand: List[Dict] = []
    for r in rows:
        if r.get("mode") != want_mode:
            continue

        # scale 비교 (None 포함)
        if (r.get("scale") is None) != (want_scale is None):
            continue
        if want_scale is not None and r.get("scale") != want_scale:
            continue

        cand.append(r)

    if not cand:
        return None

    cand.sort(
        key=lambda x: (
            x.get("rank_by_accuracy", 999999),
            -float(x.get("accuracy", -1.0)),
        )
    )
    best = cand[0]
    cp = best.get("case_path")
    if not cp:
        return None
    return str(Path(cp))  # 경로 normalize


def _infer_out_dim(model_dir: str) -> int | None:
    """체크포인트에서 classifier bias/weight를 찾아 out_features를 추론."""
    import torch
    from pathlib import Path

    ckpts = [
        Path(model_dir) / "pytorch_model.bin",
        Path(model_dir) / "pytorch_model.bin.index.json",  # sharded면 index만 있을 수도
        Path(model_dir) / "model.safetensors",
        Path(model_dir) / "pytorch_model.safetensors",
    ]

    state = None
    # 1) bin 우선 시도
    if ckpts[0].exists():
        try:
            state = torch.load(str(ckpts[0]), map_location="cpu")
        except Exception:
            state = None

    # 2) safetensors 시도
    if state is None:
        try:
            from safetensors.torch import load_file
            for p in ckpts[2:]:
                if p.exists():
                    state = load_file(str(p))
                    break
        except Exception:
            state = None

    if state is None:
        return None

    # 흔한 키들 검사
    candidate_keys = [
        "classifier.bias",
        "classifier.out_proj.bias",  # RoBERTa 일부 구현
        "score.bias",                # Distil* 변형 등
    ]
    for k in candidate_keys:
        if k in state:
            return int(state[k].shape[0])

    # bias가 없으면 weight로 추론
    candidate_keys_w = [
        "classifier.weight",
        "classifier.out_proj.weight",
        "score.weight",
    ]
    for k in candidate_keys_w:
        if k in state:
            return int(state[k].shape[0])

    return None


def load_model(model_dir: str):
    if not os.path.isdir(model_dir):
        print(f"[SKIP] no model dir at {model_dir}")
        return None, None

    # 체크포인트를 먼저 검사해서 out_dim 추론
    out_dim = _infer_out_dim(model_dir)

    # config 불러와서 필요 시 num_labels 수정
    cfg = AutoConfig.from_pretrained(model_dir)
    if out_dim is not None and getattr(cfg, "num_labels", None) != out_dim:
        cfg.num_labels = out_dim

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=cfg,                    # 우리가 수정한 num_labels 반영
        ignore_mismatched_sizes=False  # 이제는 맞아야 함
    )
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tok, model


@torch.no_grad()
def batched_logits(model, **enc):
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}
    return model(**enc).logits.detach().cpu()


# ----------------------------------------------------------------------
# 라벨 정규화
# ----------------------------------------------------------------------

def nli_norm(raw: str) -> str:
    """
    CB, RTE, AX-b, AX-g용 label 정규화.
    SuperGLUE에서 요구하는 문자열은:
      - entailment
      - contradiction
      - neutral
    (AX-b/g는 not_entailment도 있음)
    """
    r = str(raw).strip().lower()
    table = {
        "label_0": "entailment",
        "label_1": "contradiction",
        "label_2": "neutral",
        "entail": "entailment",
        "entailment": "entailment",
        "contradict": "contradiction",
        "contradiction": "contradiction",
        "neutral": "neutral",
        "not_entailment": "not_entailment",
    }
    return table.get(r, r)


def bool_norm(raw) -> bool:
    """
    BoolQ, WiC, WSC 용.
    id2label 이 'LABEL_0', 'LABEL_1' 이거나
    true/false 스트링일 수도 있어서 대충 긍정 계열을 True로 묶음.
    """
    if isinstance(raw, bool):
        if (raw): return "True"
        elif (not raw): return "False"
    r = str(raw).strip().lower()
    return "True" if (r in {"true", "1", "yes", "y", "positive", "entailment"}) else "False"


# ----------------------------------------------------------------------
# 태스크 실행
# ----------------------------------------------------------------------

def run_task(task: str, model_dir: str, mode_slug: Optional[str]) -> None:
    if task not in HF_CONFIG:
        print(f"[SKIP] unknown task: {task}")
        return
    if task not in CLASSIFICATION_TASKS:
        print(f"[SKIP] {task}: separate pipeline needed (MultiRC/ReCoRD).")
        return

    tok, model = load_model(model_dir)
    if tok is None:
        return

    ds = load_dataset("super_glue", HF_CONFIG[task])
    test = ds["test"]

    ensure_dir(OUT_DIR)

    # 출력 파일 이름: 예) CB-mix-1.jsonl
    if mode_slug is None:
        out_name = f"{task}.jsonl"
    else:
        out_name = f"{task}-{mode_slug}.jsonl"
    out_path = OUT_DIR / out_name

    id2label = getattr(model.config, "id2label", None)

    def write(rows):
        with out_path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 시작할 때 파일 비우기
    out_path.write_text("", encoding="utf-8")

    # ----------------- 태스크별 인퍼런스 -----------------

    if task in {"RTE", "AX-b", "AX-g", "CB"}:
        a = test["premise"]
        b = test["hypothesis"]
        idxs = test["idx"]
        for i in range(0, len(test), BATCH):
            enc = tok(
                a[i:i + BATCH],
                b[i:i + BATCH],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            pred = batched_logits(model, **enc).argmax(-1).tolist()
            rows = []
            for j, p in enumerate(pred):
                raw = id2label[p] if id2label else str(p)
                rows.append({
                    "idx": int(idxs[i + j]),
                    "label": nli_norm(raw),
                })
            write(rows)

    elif task == "BoolQ":
        q = ["Question: " + x for x in test["question"]]
        psg = test["passage"]
        idxs = test["idx"]
        for i in range(0, len(test), BATCH):
            enc = tok(
                q[i:i + BATCH],
                psg[i:i + BATCH],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            pred = batched_logits(model, **enc).argmax(-1).tolist()
            rows = []
            for j, p in enumerate(pred):
                raw = id2label[p] if id2label else str(p)
                rows.append({
                    "idx": int(idxs[i + j]),
                    "label": bool_norm(raw),
                })
            write(rows)

    elif task == "WiC":
        s1 = test["sentence1"]
        s2 = test["sentence2"]
        idxs = test["idx"]
        for i in range(0, len(test), BATCH):
            enc = tok(
                s1[i:i + BATCH],
                s2[i:i + BATCH],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            pred = batched_logits(model, **enc).argmax(-1).tolist()
            rows = []
            for j, p in enumerate(pred):
                raw = id2label[p] if id2label else str(p)
                rows.append({
                    "idx": int(idxs[i + j]),
                    "label": bool_norm(raw),
                })
            write(rows)

    elif task == "WSC":
        text = test["text"]
        idxs = test["idx"]
        for i in range(0, len(test), BATCH):
            enc = tok(
                text[i:i + BATCH],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            pred = batched_logits(model, **enc).argmax(-1).tolist()
            rows = []
            for j, p in enumerate(pred):
                raw = id2label[p] if id2label else str(p)
                rows.append({
                    "idx": int(idxs[i + j]),
                    "label": bool_norm(raw),
                })
            write(rows)

    elif task == "COPA":
        prem = test["premise"]; c1 = test["choice1"]; c2 = test["choice2"]; idxs = test["idx"]

        # id2label을 이용해 "긍정(정답) 클래스" 인덱스를 추정 (2-logit일 때)
        pos_idx = 1  # 기본값
        if id2label:
            # entailment / true / LABEL_1 쪽을 우선으로 찾기
            key = None
            for i, name in id2label.items():
                n = str(name).lower()
                if n in ("entailment", "true", "label_1", "yes", "positive"):
                    key = i; break
            if key is not None:
                try:
                    pos_idx = int(key)
                except Exception:
                    pass

        def score_batch(p_list, c_list):
            enc = tok(p_list, c_list, padding=True, truncation=True,
                    max_length=256, return_tensors="pt")
            logits = batched_logits(model, **enc)  # shape: (B, C)
            # C==1 이면 그대로 점수로 사용, C>=2면 "긍정 클래스" 로짓 사용
            if logits.ndim == 2 and logits.shape[-1] == 1:
                return logits[:, 0]
            elif logits.ndim == 2 and logits.shape[-1] >= 2:
                return logits[:, pos_idx]
            else:
                # 이상 케이스 방어
                return logits.view(logits.size(0), -1).max(-1).values

        for i in range(0, len(test), BATCH):
            s1 = score_batch(prem[i:i+BATCH], c1[i:i+BATCH])
            s2 = score_batch(prem[i:i+BATCH], c2[i:i+BATCH])
            rows = []
            for j, better2 in enumerate((s2 > s1).tolist()):
                rows.append({"idx": int(idxs[i+j]), "label": 1 if better2 else 0})
            write(rows)


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------

def main():
    ensure_dir(OUT_DIR)

    for task in REQUIRED_TASKS:
        rows = load_summary_rows(task)
        if not rows:
            continue

        for mode_slug in REQUIRED_MODES:
            base_mode, scale = parse_mode_slug(mode_slug)
            case_path = pick_best_case_path(rows, base_mode, scale)
            if not case_path:
                print(f"[{task}][{mode_slug}] no matching case_path, skip")
                continue

            print(f"[RUN] {task} / {mode_slug} -> {case_path}")
            run_task(task, case_path, mode_slug)

    print("\n[Done] All JSONL written to:", OUT_DIR)


if __name__ == "__main__":
    main()
