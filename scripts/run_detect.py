import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tracking.detect import YOLODetector


def resolve_sequence_dir(data_dir: Path, seq_name: str) -> Path:
    # Match exact directory first
    direct = data_dir / seq_name
    if direct.is_dir():
        return direct

    # Match prefixes, prioritizing FRCNN detector directory
    matches = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(seq_name)])
    if not matches:
        raise FileNotFoundError(f"Could not locate sequence directory for '{seq_name}' in {data_dir}")

    frcnn_matches = [d for d in matches if "FRCNN" in d.name]
    return frcnn_matches[0] if frcnn_matches else matches[0]


def ensure_frames_downloaded(seq_dir: Path) -> List[Path]:
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if frames:
        return frames

    print(f"Frames not found in {img_dir}. Downloading frames from HuggingFace dataset mirror...")
    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    repo_seq_pattern = f"train/{seq_dir.name}/img1/*"

    snapshot_download(
        repo_id="Morrison1025/MOT17",
        repo_type="dataset",
        allow_patterns=[repo_seq_pattern],
        local_dir=str(seq_dir.parent.parent),
        max_workers=16,
    )

    frames = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not frames:
        raise RuntimeError(f"Failed to obtain frames for sequence {seq_dir.name}")

    return frames


def run_detection_for_sequence(
    seq_name: str,
    seq_dir: Path,
    detector: YOLODetector,
    output_path: Path,
    meta_path: Path,
    force: bool = False,
) -> Dict[str, float]:
    # Check cache
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        lines = [line.strip() for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elapsed_sec = 0.0
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                elapsed_sec = float(meta.get("elapsed_sec", 0.0))
            except Exception:
                elapsed_sec = 0.0

        print(f"[CACHE HIT] {seq_name}: loaded {len(lines)} detections from {output_path}")
        return {
            "sequence": seq_name,
            "total_frames": 0,
            "total_detections": len(lines),
            "elapsed_sec": elapsed_sec,
            "cached": True,
        }

    frames = ensure_frames_downloaded(seq_dir)
    total_frames = len(frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running detection on {seq_name} ({total_frames} frames)...")
    out_lines: List[str] = []
    start_time = time.perf_counter()

    for idx, frame_path in enumerate(frames, start=1):
        # Extract frame index from file stem if numeric (e.g. 000001 -> 1)
        try:
            frame_num = int(frame_path.stem)
        except ValueError:
            frame_num = idx

        detections = detector.detect_frame(frame_path)
        for det in detections:
            x, y, w, h, score, _ = det
            out_lines.append(
                f"{frame_num},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n"
            )

    elapsed = time.perf_counter() - start_time
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    output_path.write_text("".join(out_lines), encoding="utf-8")

    meta_info = {
        "sequence": seq_name,
        "sequence_dir": str(seq_dir),
        "total_frames": total_frames,
        "total_detections": len(out_lines),
        "elapsed_sec": float(elapsed),
        "fps": float(fps),
        "weights": detector.weights,
        "conf": detector.conf,
        "imgsz": detector.imgsz,
        "cached": False,
    }
    meta_path.write_text(json.dumps(meta_info, indent=2), encoding="utf-8")

    return meta_info


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run object detector on MOT sequences."
    )
    parser.add_argument(
        "--seq",
        nargs="+",
        required=True,
        help="Sequence name(s) to process (e.g. MOT17-04 MOT17-02 MOT17-09)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path or name of detector model weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence score threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution for inference",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Base directory containing sequence folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dets"),
        help="Directory to store MOT-formatted detection files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (e.g. cuda:0 or cpu)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running detection even if cached file exists",
    )

    args = parser.parse_args(argv)

    detector = YOLODetector(
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    results: List[Dict[str, float]] = []

    for seq_item in args.seq:
        seq_dir = resolve_sequence_dir(args.data_dir, seq_item)
        out_file = args.output_dir / f"{seq_item}.txt"
        meta_file = args.output_dir / f"{seq_item}_meta.json"

        info = run_detection_for_sequence(
            seq_name=seq_item,
            seq_dir=seq_dir,
            detector=detector,
            output_path=out_file,
            meta_path=meta_file,
            force=args.force,
        )
        results.append(info)

    # Print summary report
    print("\nDetection Summary:")
    row_fmt = "{:<15} {:<15} {:<15} {:<15}"
    print(row_fmt.format("Sequence", "Detections", "Runtime (s)", "Status"))
    print("-" * 65)
    for r in results:
        status = "Cached" if r.get("cached") else "Computed"
        print(
            row_fmt.format(
                r["sequence"],
                str(r["total_detections"]),
                f"{r['elapsed_sec']:.2f}",
                status,
            )
        )
    print("-" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
