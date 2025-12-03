# run_all.py
import subprocess
from pathlib import Path
from datetime import datetime
from collections import deque

CONFIG_DIR = Path("train/configs")
TRAIN_ENTRY = Path("train.train_biosses")  # 또는 모듈로 실행하려면 '-m', 'train.train_one' 사용
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

    cmd = [
        "uv", "run", "--extra", "cu124",
        "python", "-m", str(TRAIN_ENTRY),
        "--config-name", config_name,
    ]

    last_lines = deque(maxlen=10)

    with log_file.open("w", encoding="utf-8") as lf:
        try:
            # 실시간 스트리밍용 Popen
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # 한 줄씩 읽어서 바로 파일/콘솔로 출력
            assert proc.stdout is not None
            for line in proc.stdout:
                lf.write(line)
                lf.flush()          # 실시간으로 파일에 찍히게
                # print(line, end="") # 콘솔에도 바로바로 출력
                last_lines.append(line.rstrip("\n"))

            # 프로세스 종료 코드
            rc = proc.wait(timeout=TIMEOUT) if TIMEOUT is not None else proc.wait()

        except subprocess.TimeoutExpired as e:
            lf.write(f"[TIMEOUT] after {e.timeout} seconds\n")
            print(f"[TIMEOUT] {config_name} after {e.timeout} seconds")
            return config_name, -1, f"[TIMEOUT] {config_name}"

        lf.write(f"\n[RETURN CODE] {rc}\n")

    # 마지막 10줄만 요약해서 다시 보여주고 싶으면:
    tail = "\n".join(last_lines)
    print(tail)
    print(f"[RESULT] {config_name} rc={rc}")
    return config_name, rc, None


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
