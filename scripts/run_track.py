import argparse
from pathlib import Path
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run object tracker on MOT sequences."
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["sort", "deepsort"],
        default="sort",
        help="Tracking algorithm to run",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Path to MOT sequences directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tracks"),
        help="Path to store tracking output files",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional list of sequence names to process",
    )
    args = parser.parse_args(argv)
    print(f"Tracking CLI configured: tracker={args.tracker}, data_dir={args.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
