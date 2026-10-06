import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Sequence
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tracking.io import format_mot_row, load_mot_detections
from src.tracking.sort import Sort


def find_detection_file(
    seq_name: str, dets_dir: Path, data_dir: Path
) -> Path:
    direct = dets_dir / f"{seq_name}.txt"
    if direct.is_file():
        return direct

    # Check for prefix match in results/dets
    matches = list(dets_dir.glob(f"{seq_name}*.txt"))
    if matches:
        return matches[0]

    # Fallback to data sequence det.txt
    seq_dir_matches = list(data_dir.glob(f"{seq_name}*"))
    for s_dir in seq_dir_matches:
        det_file = s_dir / "det" / "det.txt"
        if det_file.is_file():
            return det_file

    raise FileNotFoundError(
        f"No detection file found for sequence '{seq_name}' in {dets_dir} or {data_dir}"
    )


def run_sort_on_sequence(
    seq_name: str,
    det_file: Path,
    output_path: Path,
    meta_path: Path,
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
) -> Dict[str, float]:
    dets_by_frame = load_mot_detections(det_file)
    if not dets_by_frame:
        print(f"Warning: No detections found in {det_file}")
        max_frame = 0
    else:
        max_frame = max(dets_by_frame.keys())

    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    tracker.reset_id_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    out_lines: List[str] = []
    total_active_tracks = set()
    start_time = time.perf_counter()

    for frame in range(1, max_frame + 1):
        frame_dets = dets_by_frame.get(frame, np.empty((0, 5), dtype=float))
        track_outputs = tracker.update(frame_dets)

        for trk in track_outputs:
            x1, y1, x2, y2, trk_id, conf = trk[:6]
            total_active_tracks.add(int(trk_id))
            out_lines.append(
                format_mot_row(frame, int(trk_id), np.array([x1, y1, x2, y2]), conf)
            )

    elapsed = time.perf_counter() - start_time
    fps = max_frame / elapsed if elapsed > 0 else 0.0

    output_path.write_text("".join(out_lines), encoding="utf-8")

    meta_info = {
        "sequence": seq_name,
        "tracker": "sort",
        "total_frames": max_frame,
        "unique_tracks": len(total_active_tracks),
        "total_instances": len(out_lines),
        "elapsed_sec": float(elapsed),
        "fps": float(fps),
        "max_age": max_age,
        "min_hits": min_hits,
        "iou_threshold": iou_threshold,
    }
    meta_path.write_text(json.dumps(meta_info, indent=2), encoding="utf-8")

    return meta_info


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
        "--seq",
        nargs="+",
        default=["MOT17-04", "MOT17-02", "MOT17-09"],
        help="Sequence name(s) to process",
    )
    parser.add_argument(
        "--dets-dir",
        type=Path,
        default=Path("results/dets"),
        help="Directory containing detection files",
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
        help="Base path to store tracking output files",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Maximum frames to retain track without association",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum hits before track is confirmed",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU match threshold for track association",
    )

    args = parser.parse_args(argv)

    if args.tracker != "sort":
        print(f"Tracker '{args.tracker}' is not implemented yet.", file=sys.stderr)
        return 1

    tracker_dir = args.output_dir / args.tracker

    summary_rows: List[Dict[str, float]] = []

    for seq_item in args.seq:
        det_file = find_detection_file(seq_item, args.dets_dir, args.data_dir)
        out_file = tracker_dir / f"{seq_item}.txt"
        meta_file = tracker_dir / f"{seq_item}_meta.json"

        print(f"Running {args.tracker} on {seq_item} using detections from {det_file}...")
        info = run_sort_on_sequence(
            seq_name=seq_item,
            det_file=det_file,
            output_path=out_file,
            meta_path=meta_file,
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
        )
        summary_rows.append(info)
        print(
            f"[{args.tracker.upper()}] {seq_item}: "
            f"{info['total_frames']} frames, {info['unique_tracks']} tracks, "
            f"{info['total_instances']} instances in {info['elapsed_sec']:.3f}s ({info['fps']:.1f} FPS) "
            f"-> {out_file}"
        )

    print("\nTracking Run Summary:")
    row_fmt = "{:<12} {:<10} {:<12} {:<12} {:<12} {:<10}"
    print(row_fmt.format("Sequence", "Frames", "Unique IDs", "Instances", "Runtime (s)", "FPS"))
    print("-" * 75)
    for row in summary_rows:
        print(
            row_fmt.format(
                row["sequence"],
                str(row["total_frames"]),
                str(row["unique_tracks"]),
                str(row["total_instances"]),
                f"{row['elapsed_sec']:.3f}",
                f"{row['fps']:.1f}",
            )
        )
    print("-" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
