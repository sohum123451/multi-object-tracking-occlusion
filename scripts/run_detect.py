import argparse
from pathlib import Path
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run object detector on MOT sequence frames."
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
        default=Path("results/detections"),
        help="Path to store detection text files",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="yolov8n.pt",
        help="Detector model weights path or identifier",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Confidence score threshold",
    )
    args = parser.parse_args(argv)
    print(f"Detector CLI configured: data_dir={args.data_dir}, output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
