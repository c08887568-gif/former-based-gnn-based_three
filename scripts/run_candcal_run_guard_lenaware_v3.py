import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    script = PROJECT_ROOT / "experiments" / "train_candcal_run_guard_lenaware_v3.py"
    command = [sys.executable, str(script)]
    raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
