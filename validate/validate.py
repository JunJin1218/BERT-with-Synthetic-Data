# summarize_results.py
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

ROOT = Path("models")  # models 루트

MODES = ["original", "mix", "synthetic"]
SUMMARY_FILENAME = "summary_by_accuracy.jsonl"


def read_results_json(path: Path) -> Optional[Dict[str, Any]]:
    f = path / "results.json"
    if not f.exists():
        return None
    try:
        with f.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def collect_cases_for_task(task_dir: Path) -> List[Dict[str, Any]]:
    """
    task_dir: models/super_glue-{task}
    디렉토리 구조 가정 (변경 후):
      - original/{time}/            -> results.json (scale 없음)
      - mix/{scale}/{time}/         -> results.json
      - synthetic/{scale}/{time}/   -> results.json
    """
    cases: List[Dict[str, Any]] = []

    for mode in MODES:
        mode_dir = task_dir / mode
        if not mode_dir.exists():
            continue

        # original: 아래에 time 디렉토리들
        if mode == "original":
            for time_dir in sorted(mode_dir.iterdir()):
                if not time_dir.is_dir():
                    continue
                res = read_results_json(time_dir)
                if res:
                    cases.append(
                        build_record(
                            res,
                            task_dir=task_dir,
                            mode=mode,
                            scale=None,
                            base_dir=time_dir,
                        )
                    )
            continue

        # mix/synthetic: scale 디렉토리 → 그 아래 time 디렉토리
        for scale_dir in sorted(mode_dir.iterdir()):
            if not scale_dir.is_dir():
                continue
            try:
                scale = int(scale_dir.name)
            except ValueError:
                # scale 이름이 숫자가 아닌 폴더는 무시
                continue

            for time_dir in sorted(scale_dir.iterdir()):
                if not time_dir.is_dir():
                    continue
                res = read_results_json(time_dir)
                if res:
                    cases.append(
                        build_record(
                            res,
                            task_dir=task_dir,
                            mode=mode,
                            scale=scale,
                            base_dir=time_dir,
                        )
                    )

    return cases


def build_record(
    res: Dict[str, Any],
    task_dir: Path,
    mode: str,
    scale: Optional[int],
    base_dir: Path,
) -> Dict[str, Any]:
    # 안전 파싱
    vr = res.get("validation_results", {})
    bm = res.get("best_model", {})
    cfg = res.get("config", {})

    accuracy = vr.get("eval_accuracy")
    f1_macro = vr.get("eval_f1_macro")
    # best_epoch가 null일 수 있으니, 없으면 validation_results의 epoch라도 기록
    best_epoch = bm.get("epoch", None)
    if best_epoch is None:
        best_epoch = vr.get("epoch")

    record: Dict[str, Any] = {
        "task_dir": str(task_dir.as_posix()),  # models/super_glue-{task}
        "task_name": res.get("task_name") or f"{task_dir.name}",
        "mode": mode,
        "scale": scale,  # original이면 None
        "timestamp": res.get("timestamp"),
        # 정렬/리포팅 핵심
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "best_epoch": best_epoch,
        # 참고용 메타
        "best_metric_name": (bm.get("metric_name") or cfg.get("metric_for_best_model")),
        "best_metric_value": bm.get("metric_value"),
        "best_checkpoint": bm.get("checkpoint"),
        "config_metric_for_best": cfg.get("metric_for_best_model"),
        # 해당 케이스 디렉토리 (…/original/{time}, …/mix/{scale}/{time} 등)
        "path": str(base_dir.as_posix()),
    }
    return record


def _score_for_best(rec: Dict[str, Any]):
    """
    (mode, scale) 그룹 안에서 '가장 좋은 time' 고를 때 사용할 스코어.
    1순위: accuracy (None < 값 있는 것, 값 큰 것 우선)
    2순위: f1_macro (있으면 큰 것 우선)
    3순위: timestamp (있으면 큰/나중 것 우선) – 필요하면 타이브레이크 용
    """
    acc = rec.get("accuracy")
    f1 = rec.get("f1_macro")
    ts = rec.get("timestamp")

    # None-safe 처리
    acc_present = acc is not None
    f1_present = f1 is not None
    ts_present = ts is not None

    acc_val = -1e9 if acc is None else float(acc)
    f1_val = -1e9 if f1 is None else float(f1)

    # timestamp는 문자열이지만, 같은 mode/scale 내에서
    # 그냥 사전순 비교해도 "20251202_012317" 같은 포맷이면 시간 순서와 같다고 가정
    ts_val = ts if ts is not None else ""

    return (
        acc_present,
        acc_val,
        f1_present,
        f1_val,
        ts_present,
        ts_val,
    )


def select_best_per_mode_scale(
    cases: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    각 (mode, scale) 조합에 대해 여러 time 실행 중 '가장 좋은' run 하나만 선택.
    """
    grouped: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = defaultdict(list)

    for rec in cases:
        key = (rec["mode"], rec["scale"])
        grouped[key].append(rec)

    best_records: List[Dict[str, Any]] = []
    for (mode, scale), recs in grouped.items():
        best = max(recs, key=_score_for_best)
        best_records.append(best)

    return best_records


def write_summary(task_dir: Path, best_cases: List[Dict[str, Any]]) -> Path:
    """
    best_cases: 이미 각 (mode, scale)별로 '가장 좋은 time'만 골라진 리스트.
    여기서 accuracy 기준 내림차순으로 rank 매기고 summary_by_accuracy.jsonl에 기록.
    """
    # accuracy 기준 내림차순 정렬(없으면 뒤로)
    def sort_key(x: Dict[str, Any]):
        acc = x.get("accuracy")
        if isinstance(acc, (int, float)):
            return acc
        return -1e9

    sorted_cases = sorted(best_cases, key=sort_key, reverse=True)

    out_path = task_dir / SUMMARY_FILENAME
    with out_path.open("w", encoding="utf-8") as fw:
        for rank, rec in enumerate(sorted_cases, start=1):
            payload = {
                "rank_by_accuracy": rank,
                "task": rec["task_name"],
                "mode": rec["mode"],
                "scale": rec["scale"],
                "accuracy": rec["accuracy"],
                "f1_macro": rec["f1_macro"],
                "best_epoch": rec["best_epoch"],
                "best_metric_name": rec["best_metric_name"],
                "best_metric_value": rec["best_metric_value"],
                "checkpoint": rec["best_checkpoint"],
                "timestamp": rec["timestamp"],  # <- 여기가 "어떤 time이 best인지"에 해당
                "case_path": rec["path"],       # e.g. models/super_glue-cb/mix/2/20251202_012317
            }
            fw.write(json.dumps(payload, ensure_ascii=False))
            fw.write("\n")
    return out_path


def _fmt_float(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return "None"


def main():
    # models/super_glue-* 디렉토리들 순회
    task_dirs = sorted(
        [p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("super_glue-")]
    )
    if not task_dirs:
        print("[WARN] No task directories found under 'models/'. Expected: models/super_glue-{task}")
        return

    for tdir in task_dirs:
        cases = collect_cases_for_task(tdir)
        print("="*30)
        print(tdir)
        if not cases:
            print(f"[INFO] Skip (no results): {tdir}")
            continue

        # 1) 각 (mode, scale)에 대해 best time 하나씩 고르기
        best_cases = select_best_per_mode_scale(cases)

        # 2) 그 best 들만 가지고 accuracy 기준 rank 매기고 summary 파일 작성
        out_path = write_summary(tdir, best_cases)
        print(f"[OK] Wrote summary (accuracy-desc, best per mode/scale): {out_path}")

        # 콘솔 로그:
        print("  Best per mode/scale:")
        # (mode, scale) 순으로 보기 좋게 정렬
        for rec in sorted(
            best_cases,
            key=lambda r: (r["mode"], -1 if r["scale"] is None else r["scale"]),
        ):
            time_dir_name = Path(rec["path"]).name  # 마지막 폴더 이름이 time
            print(
                f"   - mode={rec['mode']:<9} "
                f"scale={str(rec['scale']):>4} "
                f"time={time_dir_name:<15} "
                f"acc={_fmt_float(rec['accuracy'])} "
                f"f1={_fmt_float(rec['f1_macro'])} "
                f"epoch={rec['best_epoch']}"
            )

        # accuracy 기준 상위 5개 프리뷰 (이미 best만 모아놨으니 사실상 top-5 mode/scale)
        print("  Top-5 preview (by accuracy, best per mode/scale):")
        preview = sorted(
            best_cases,
            key=lambda x: (x.get("accuracy") is not None, x.get("accuracy") or -1e9),
            reverse=True,
        )[:5]
        for i, rec in enumerate(preview, 1):
            print(
                f"   {i:>2}. {rec['mode']:<9} "
                f"scale={str(rec['scale']):>4}  "
                f"acc={_fmt_float(rec['accuracy'])}  "
                f"f1={_fmt_float(rec['f1_macro'])}  "
                f"epoch={rec['best_epoch']}  "
                f"time={Path(rec['path']).name}"
            )

if __name__ == "__main__":
    main()