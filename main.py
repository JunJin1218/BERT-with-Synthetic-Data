# run_all.py
import subprocess
from pathlib import Path
from datetime import datetime

CONFIG_DIR = Path("train/configs")
TRAIN_ENTRY = Path("train.train_one")  # 또는 모듈로 실행하려면 '-m', 'train.train_one' 사용
LOG_DIR = Path("logs/train_one_runs")
TIMEOUT = None  # 초 단위. 원하면 7200 같이 제한

def list_configs(config_dir: Path):
    return sorted([*config_dir.glob("*.yaml"), *config_dir.glob("*.yml")])

def run_one(config_path: Path):
    config_name = config_path.stem  # 파일명에서 확장자 제거 => --config-name
    log_dir = LOG_DIR / config_name
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{ts}.log"

    print("=" * 80)
    print(f"[RUN] config: {config_name}")
    print(f"      path  : {config_path}")
    print(f"      log   : {log_file}")
    print("=" * 80)

    # uv로 실행 (권장). 모듈 방식으로 실행하고 싶으면 아래 주석 해제
    cmd = [
        "uv", "run", "--extra", "cu124",
        "python", "-m", str(TRAIN_ENTRY),
        "--config-name", config_name,
    ]

    # 모듈로 실행: cmd = ["uv","run","--extra","cu124","python","-m","train.train_one","--config-path",str(CONFIG_DIR),"--config-name",config_name]

    # 부모 프로세스의 CWD 기준으로 실행.
    # 자식은 Hydra 때문에 작업 디렉토리가 바뀔 수 있으니, 자식 코드에서 to_absolute_path로 경로 고정 추천.
    with log_file.open("w", encoding="utf-8") as lf:
        try:
            cp = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            lf.write(f"[TIMEOUT] after {e.timeout} seconds\n")
            return config_name, -1, f"[TIMEOUT] {config_name}"

        # 로그 파일 저장
        lf.write(cp.stdout if cp.stdout else "")
        lf.write(f"\n[RETURN CODE] {cp.returncode}\n")

    # 콘솔 요약
    tail = "\n".join((cp.stdout or "").splitlines()[-10:])  # 마지막 10줄만 콘솔 요약
    print(tail)
    print(f"[RESULT] {config_name} rc={cp.returncode}")
    return config_name, cp.returncode, None

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    configs = list_configs(CONFIG_DIR)
    if not configs:
        print(f"No config files found in {CONFIG_DIR.resolve()}")
        return

    results = []
    for cfg in configs:
        results.append(run_one(cfg))

    print("\n" + "#" * 80)
    print("# SUMMARY")
    print("#" * 80)
    ok, fail = 0, 0
    for name, rc, err in results:
        status = "OK" if rc == 0 else "FAIL"
        print(f"{status:4} - {name}")
        if rc == 0:
            ok += 1
        else:
            fail += 1

    print("-" * 80)
    print(f"Total: {len(results)} | OK: {ok} | FAIL: {fail}")

if __name__ == "__main__":
    main()
