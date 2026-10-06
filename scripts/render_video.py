import argparse
from pathlib import Path
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render tracking bounding boxes and trajectory trails onto video frames."
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Path to sequence directory containing img1/",
    )
    parser.add_argument(
        "--tracks-file",
        type=Path,
        required=True,
        help="Path to tracking result text file",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        required=True,
        help="Path to output rendered video file",
    )
    args = parser.parse_args(argv)
    print(f"Video render CLI configured: output_video={args.output_video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
