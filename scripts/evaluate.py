import argparse
from pathlib import Path
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate tracking predictions against ground truth."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Path to MOT ground truth sequences",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/tracks"),
        help="Path to tracker results directory",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["sort", "deepsort"],
        help="Trackers to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Path to write evaluation outputs",
    )
    args = parser.parse_args(argv)
    print(f"Evaluation CLI configured: data_dir={args.data_dir}, results_dir={args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
