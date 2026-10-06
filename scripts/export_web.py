import argparse
from pathlib import Path
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export tracking metrics and visual demonstrations to web directory."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to results directory",
    )
    parser.add_argument(
        "--web-dir",
        type=Path,
        default=Path("web"),
        help="Path to web directory",
    )
    args = parser.parse_args(argv)
    print(f"Web export CLI configured: results_dir={args.results_dir}, web_dir={args.web_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
