import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run CANDCAL Run Guard v2 experiments.")
    parser.add_argument("--exp", choices=["ALL", "WEIGHTED", "ADAPTDELTA", "WEIGHTED_ADAPTDELTA"], default="ALL")
    args = parser.parse_args()
    script = PROJECT_ROOT / "experiments" / "train_candcal_run_guard_v2_3runs.py"
    command = [sys.executable, str(script)]
    if args.exp != "ALL":
        command.extend(["--exp", args.exp])
    raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
