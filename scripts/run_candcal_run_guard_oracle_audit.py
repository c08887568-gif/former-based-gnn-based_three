import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    script = PROJECT_ROOT / "experiments" / "candcal_run_guard_oracle_audit.py"
    raise_code = subprocess.call([sys.executable, str(script)], cwd=PROJECT_ROOT)
    raise SystemExit(raise_code)


if __name__ == "__main__":
    main()
