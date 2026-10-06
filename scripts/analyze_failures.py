import argparse
from pathlib import Path
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze tracking failure modes including ID switches and fragmentation."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Path to MOT sequences directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/tracks"),
        help="Path to tracker results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/failure_analysis"),
        help="Path to write analysis reports",
    )
    args = parser.parse_args(argv)
    print(f"Failure analysis CLI configured: results_dir={args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
