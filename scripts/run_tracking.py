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

from src.data.mot_loader import Detection, MOTSequence, load_sequence
from src.tracker.byte_tracker import ByteTracker
from src.tracker.sort_tracker import SortTracker



def format_mot_line(
    frame: int,
    track_id: int,
    tlwh: np.ndarray,
    score: float = 1.0,
) -> str:
    x, y, w, h = tlwh
    return f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n"


def run_tracker_on_sequence(
    tracker_name: str,
    sequence: MOTSequence,
    output_path: Path,
    timing_path: Path,
    det_thresh: float = 0.5,
    track_thresh: float = 0.5,
    track_buffer: int = 30,
) -> Dict[str, float]:
    if tracker_name == "sort":
        tracker = SortTracker(det_thresh=det_thresh, max_age=track_buffer)
    elif tracker_name == "bytetrack":
        tracker = ByteTracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
        )
    else:
        raise ValueError(f"Unknown tracker name: {tracker_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    detections_by_frame = sequence.get_detections_by_frame()
    total_frames = sequence.seq_length

    lines: List[str] = []
    start_time = time.perf_counter()

    for frame in range(1, total_frames + 1):
        dets = detections_by_frame.get(frame, [])
        if len(dets) > 0:
            arr = np.zeros((len(dets), 5), dtype=float)
            for idx, d in enumerate(dets):
                arr[idx, :4] = d.bbox_tlbr
                arr[idx, 4] = d.score
        else:
            arr = np.empty((0, 5), dtype=float)

        tracks = tracker.update(arr)

        for trk in tracks:
            lines.append(format_mot_line(frame, trk.track_id, trk.tlwh, trk.score))

    elapsed = time.perf_counter() - start_time
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    output_path.write_text("".join(lines), encoding="utf-8")

    timing_info = {
        "sequence": sequence.name,
        "tracker": tracker_name,
        "total_frames": total_frames,
        "elapsed_sec": float(elapsed),
        "fps": float(fps),
    }
    timing_path.write_text(json.dumps(timing_info, indent=2), encoding="utf-8")

    return timing_info


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run object tracker on MOT sequences.")
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["sort", "bytetrack"],
        choices=["sort", "bytetrack"],
        help="Trackers to execute",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Directory containing MOT sequence directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for tracker results",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional sequence filter",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.5,
        help="Detection threshold for SORT",
    )
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.5,
        help="Primary track threshold for ByteTrack",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="Lost track buffer length in frames",
    )

    args = parser.parse_args(argv)

    if not args.data_dir.exists():
        print(f"Data directory {args.data_dir} does not exist.")
        return 1

    seq_dirs = [d for d in args.data_dir.iterdir() if d.is_dir()]
    if args.sequences:
        seq_dirs = [d for d in seq_dirs if d.name in args.sequences]

    if not seq_dirs:
        print(f"No matching sequence directories found in {args.data_dir}")
        return 1

    for seq_dir in sorted(seq_dirs):
        sequence = load_sequence(seq_dir)
        print(f"Loaded sequence {sequence.name} ({sequence.seq_length} frames, {len(sequence.detections)} detections)")

        for tracker_name in args.trackers:
            out_file = args.output_dir / tracker_name / f"{sequence.name}.txt"
            time_file = args.output_dir / tracker_name / f"{sequence.name}_timing.json"

            timing = run_tracker_on_sequence(
                tracker_name=tracker_name,
                sequence=sequence,
                output_path=out_file,
                timing_path=time_file,
                det_thresh=args.det_thresh,
                track_thresh=args.track_thresh,
                track_buffer=args.track_buffer,
            )
            print(
                f"[{tracker_name.upper()}] {sequence.name}: "
                f"{timing['total_frames']} frames in {timing['elapsed_sec']:.3f}s ({timing['fps']:.1f} FPS) "
                f"-> {out_file}"
            )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
