import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tracking.deepsort import DeepSort
from src.tracking.embed import OSNetExtractor
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


def find_image_dir(seq_name: str, data_dir: Path) -> Path:
    direct = data_dir / seq_name / "img1"
    if direct.is_dir():
        return direct

    frcnn = data_dir / f"{seq_name}-FRCNN" / "img1"
    if frcnn.is_dir():
        return frcnn

    matches = list(data_dir.glob(f"{seq_name}*/img1"))
    if matches and matches[0].is_dir():
        return matches[0]

    raise FileNotFoundError(
        f"Could not find img1 directory for sequence '{seq_name}' in {data_dir}"
    )


def run_sort_on_sequence(
    seq_name: str,
    det_file: Path,
    output_path: Path,
    meta_path: Path,
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
) -> Dict[str, Any]:
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


def run_deepsort_on_sequence(
    seq_name: str,
    det_file: Path,
    img_dir: Path,
    output_path: Path,
    meta_path: Path,
    extractor: OSNetExtractor,
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    max_cosine_distance: float = 0.2,
    nn_budget: int = 100,
    gating_threshold: float = 9.4877,
    cache_embeddings: bool = True,
    embeddings_dir: Path = Path("results/embeddings"),
) -> Dict[str, Any]:
    dets_by_frame = load_mot_detections(det_file)
    if not dets_by_frame:
        print(f"Warning: No detections found in {det_file}")
        max_frame = 0
    else:
        max_frame = max(dets_by_frame.keys())

    cache_file = embeddings_dir / f"{seq_name}_{extractor.model_name}.npz"
    features_by_frame: Dict[int, np.ndarray] = {}
    is_cached = cache_file.is_file() if cache_embeddings else False

    if is_cached:
        print(f"Loading cached embeddings from {cache_file}...")
        with np.load(cache_file) as data:
            for k in data.files:
                frame_idx = int(k.replace("frame_", ""))
                features_by_frame[frame_idx] = data[k]
    else:
        embeddings_dir.mkdir(parents=True, exist_ok=True)

    tracker = DeepSort(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        max_cosine_distance=max_cosine_distance,
        nn_budget=nn_budget,
        gating_threshold=gating_threshold,
    )
    tracker.reset_id_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    out_lines: List[str] = []
    total_active_tracks = set()
    start_time = time.perf_counter()

    for frame in range(1, max_frame + 1):
        frame_dets = dets_by_frame.get(frame, np.empty((0, 5), dtype=float))

        if not is_cached:
            if len(frame_dets) > 0:
                img_path = img_dir / f"{frame:06d}.jpg"
                frame_bgr = cv2.imread(str(img_path))
                if frame_bgr is None:
                    raise FileNotFoundError(
                        f"Frame image not found or unreadable: {img_path}"
                    )
                frame_feats = extractor.extract_from_frame(
                    frame_bgr, frame_dets[:, :4]
                )
            else:
                frame_feats = np.empty((0, 512), dtype=np.float32)
            features_by_frame[frame] = frame_feats
        else:
            frame_feats = features_by_frame.get(
                frame, np.empty((0, 512), dtype=np.float32)
            )

        boxes = frame_dets[:, :4] if len(frame_dets) > 0 else np.empty((0, 4), dtype=float)
        scores = frame_dets[:, 4] if len(frame_dets) > 0 else np.empty((0,), dtype=float)

        track_outputs = tracker.update(boxes, frame_feats, scores)

        for trk in track_outputs:
            x1, y1, x2, y2, trk_id, conf = trk[:6]
            total_active_tracks.add(int(trk_id))
            out_lines.append(
                format_mot_row(frame, int(trk_id), np.array([x1, y1, x2, y2]), conf)
            )

        if not is_cached and (frame % 100 == 0 or frame == max_frame):
            print(f"  [{seq_name}] frame {frame}/{max_frame} processed...")

    if not is_cached and cache_embeddings and features_by_frame:
        np.savez_compressed(
            cache_file,
            **{f"frame_{k}": v for k, v in features_by_frame.items()},
        )

    elapsed = time.perf_counter() - start_time
    fps = max_frame / elapsed if elapsed > 0 else 0.0

    output_path.write_text("".join(out_lines), encoding="utf-8")

    meta_info = {
        "sequence": seq_name,
        "tracker": "deepsort",
        "total_frames": max_frame,
        "unique_tracks": len(total_active_tracks),
        "total_instances": len(out_lines),
        "elapsed_sec": float(elapsed),
        "fps": float(fps),
        "max_age": max_age,
        "min_hits": min_hits,
        "iou_threshold": iou_threshold,
        "max_cosine_distance": max_cosine_distance,
        "nn_budget": nn_budget,
        "gating_threshold": gating_threshold,
        "embed_model": extractor.model_name,
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
    parser.add_argument(
        "--max-cosine-distance",
        type=float,
        default=0.2,
        help="Maximum cosine distance threshold for DeepSORT appearance matching",
    )
    parser.add_argument(
        "--nn-budget",
        type=int,
        default=100,
        help="Maximum gallery size of appearance embeddings per track",
    )
    parser.add_argument(
        "--gating-threshold",
        type=float,
        default=9.4877,
        help="Mahalanobis gating threshold from Kalman state (chi-square 0.95)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="osnet_x0_25",
        help="Re-ID embedding model architecture",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Batch size for Re-ID embedding extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device for feature extractor (cpu or cuda)",
    )
    parser.add_argument(
        "--no-cache-embeddings",
        action="store_true",
        help="Disable caching of extracted appearance embeddings",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("results/embeddings"),
        help="Directory to store cached appearance embeddings",
    )

    args = parser.parse_args(argv)

    tracker_dir = args.output_dir / args.tracker

    extractor: OSNetExtractor | None = None
    if args.tracker == "deepsort":
        extractor = OSNetExtractor(
            model_name=args.embed_model,
            device=args.device,
        )

    summary_rows: List[Dict[str, Any]] = []

    for seq_item in args.seq:
        det_file = find_detection_file(seq_item, args.dets_dir, args.data_dir)
        out_file = tracker_dir / f"{seq_item}.txt"
        meta_file = tracker_dir / f"{seq_item}_meta.json"

        print(f"Running {args.tracker} on {seq_item} using detections from {det_file}...")
        if args.tracker == "sort":
            info = run_sort_on_sequence(
                seq_name=seq_item,
                det_file=det_file,
                output_path=out_file,
                meta_path=meta_file,
                max_age=args.max_age,
                min_hits=args.min_hits,
                iou_threshold=args.iou_threshold,
            )
        elif args.tracker == "deepsort":
            assert extractor is not None
            img_dir = find_image_dir(seq_item, args.data_dir)
            info = run_deepsort_on_sequence(
                seq_name=seq_item,
                det_file=det_file,
                img_dir=img_dir,
                output_path=out_file,
                meta_path=meta_file,
                extractor=extractor,
                max_age=args.max_age,
                min_hits=args.min_hits,
                iou_threshold=args.iou_threshold,
                max_cosine_distance=args.max_cosine_distance,
                nn_budget=args.nn_budget,
                gating_threshold=args.gating_threshold,
                cache_embeddings=not args.no_cache_embeddings,
                embeddings_dir=args.embeddings_dir,
            )
        else:
            raise ValueError(f"Unknown tracker: {args.tracker}")

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
