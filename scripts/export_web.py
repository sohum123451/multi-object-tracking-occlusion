import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_video import find_image_dir, load_tracks


def export_tracker_tracks_json(
    tracks_by_frame: Dict[int, List[Tuple[int, np.ndarray, float]]],
    max_frames: int,
    output_path: Path,
) -> List[Dict[str, Any]]:
    # Aggregate by track_id
    track_records: Dict[int, Dict[str, List[Any]]] = {}

    for frame in range(1, max_frames + 1):
        trks = tracks_by_frame.get(frame, [])
        for trk_id, box, conf in trks:
            foot_x = round(float((box[0] + box[2]) / 2.0), 2)
            foot_y = round(float(box[3]), 2)
            w = round(float(box[2] - box[0]), 2)
            h = round(float(box[3] - box[1]), 2)

            if trk_id not in track_records:
                track_records[trk_id] = {
                    "id": trk_id,
                    "frames": [],
                    "foot_x": [],
                    "foot_y": [],
                    "w": [],
                    "h": [],
                }
            track_records[trk_id]["frames"].append(frame)
            track_records[trk_id]["foot_x"].append(foot_x)
            track_records[trk_id]["foot_y"].append(foot_y)
            track_records[trk_id]["w"].append(w)
            track_records[trk_id]["h"].append(h)

    # Sort tracks by starting frame and length
    records_list = sorted(track_records.values(), key=lambda r: (r["frames"][0], -len(r["frames"])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records_list, f, indent=None, separators=(",", ":"))

    return records_list


def export_id_switches_json(
    switches_csv: Path,
    seq_name: str,
    max_frames: int,
    tracks_by_tracker: Dict[str, Dict[int, List[Tuple[int, np.ndarray, float]]]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    switches_list: List[Dict[str, Any]] = []
    if not switches_csv.is_file():
        return switches_list

    with switches_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sequence") != seq_name:
                continue
            frame = int(row["frame"])
            if frame > max_frames:
                continue

            tracker = row["tracker"]
            new_pred_id = int(row["new_pred_id"])
            old_pred_id = int(row["old_pred_id"])

            # Find foot position in that tracker at that frame
            foot_x = 0.0
            foot_y = 0.0
            trks = tracks_by_tracker.get(tracker, {}).get(frame, [])
            for tid, box, _ in trks:
                if tid == new_pred_id:
                    foot_x = round(float((box[0] + box[2]) / 2.0), 2)
                    foot_y = round(float(box[3]), 2)
                    break

            switches_list.append({
                "tracker": tracker,
                "frame": frame,
                "gt_id": int(row["gt_id"]),
                "other_gt_id": int(row.get("other_gt_id", -1)),
                "old_pred_id": old_pred_id,
                "new_pred_id": new_pred_id,
                "foot_x": foot_x,
                "foot_y": foot_y,
                "causes": row.get("causes", ""),
                "visibility": float(row.get("gt_visibility", 1.0)),
                "max_gt_iou": float(row.get("max_gt_iou", 0.0)),
                "speed": float(row.get("gt_speed", 0.0)),
                "cos_sim": float(row.get("cos_sim", 0.0)),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(switches_list, f, indent=2)

    return switches_list


def export_keyframes(
    img_dir: Path,
    output_dir: Path,
    max_frames: int = 300,
    step: int = 5,
    target_width: int = 480,
) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_w, orig_h = 1920, 1080

    for f_idx in range(1, max_frames + 1):
        if f_idx % step != 1 and f_idx != max_frames and f_idx != 1:
            continue
        img_path = img_dir / f"{f_idx:06d}.jpg"
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        scale = target_width / float(orig_w)
        target_h = int(round(orig_h * scale))

        small = cv2.resize(img, (target_width, target_h), interpolation=cv2.INTER_AREA)
        out_file = output_dir / f"frame_{f_idx:04d}.jpg"
        cv2.imwrite(str(out_file), small, [cv2.IMWRITE_JPEG_QUALITY, 75])

        if f_idx == 1:
            # Also save primary backdrop
            cv2.imwrite(str(output_dir.parent / "backdrop.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 75])

    return orig_w, orig_h


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export tracking data, ID switches, and video keyframes for 3D web viewer."
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="MOT17-04",
        help="Sequence name to export",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=300,
        help="Number of frames to export",
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("results/tracks"),
        help="Path to results/tracks directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Path to MOT sequences directory",
    )
    parser.add_argument(
        "--switches-csv",
        type=Path,
        default=Path("results/id_switches.csv"),
        help="Path to results/id_switches.csv",
    )
    parser.add_argument(
        "--web-public-dir",
        type=Path,
        default=Path("web/public"),
        help="Path to web/public directory",
    )
    parser.add_argument(
        "--keyframe-step",
        type=int,
        default=5,
        help="Downsampling step for video keyframes",
    )

    args = parser.parse_args(argv)

    img_dir = find_image_dir(args.seq, args.data_dir)
    sort_tracks_path = args.tracks_dir / "sort" / f"{args.seq}.txt"
    deepsort_tracks_path = args.tracks_dir / "deepsort" / f"{args.seq}.txt"

    print(f"Exporting {args.seq} data to {args.web_public_dir}...")

    # 1. Load tracker predictions
    sort_by_frame = load_tracks(sort_tracks_path)
    deepsort_by_frame = load_tracks(deepsort_tracks_path)

    # 2. Export tracks json
    sort_records = export_tracker_tracks_json(
        sort_by_frame, args.num_frames, args.web_public_dir / "tracks_sort.json"
    )
    deepsort_records = export_tracker_tracks_json(
        deepsort_by_frame, args.num_frames, args.web_public_dir / "tracks_deepsort.json"
    )
    print(f"Exported {len(sort_records)} SORT tracks and {len(deepsort_records)} DeepSORT tracks.")

    # 3. Export ID switches json
    tracks_by_tracker = {"sort": sort_by_frame, "deepsort": deepsort_by_frame}
    switches = export_id_switches_json(
        args.switches_csv,
        args.seq,
        args.num_frames,
        tracks_by_tracker,
        args.web_public_dir / "id_switches.json",
    )
    print(f"Exported {len(switches)} ID switch events for {args.seq}.")

    # 4. Export keyframes
    keyframes_dir = args.web_public_dir / "keyframes"
    orig_w, orig_h = export_keyframes(
        img_dir, keyframes_dir, max_frames=args.num_frames, step=args.keyframe_step
    )
    print(f"Exported keyframes to {keyframes_dir}")

    # 5. Export metadata
    meta = {
        "sequence": args.seq,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "num_frames": args.num_frames,
        "keyframe_step": args.keyframe_step,
        "sort_track_count": len(sort_records),
        "deepsort_track_count": len(deepsort_records),
        "switch_count": len(switches),
    }
    meta_path = args.web_public_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported metadata to {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
