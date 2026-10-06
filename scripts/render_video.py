import argparse
from collections import deque
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tracking.io import load_mot_detections


def generate_color_palette(n: int = 128) -> List[Tuple[int, int, int]]:
    np.random.seed(42)
    palette: List[Tuple[int, int, int]] = []
    for i in range(n):
        # Generate distinct hues in HSV, convert to BGR
        hue = int((i * 179 / n) % 180)
        sat = int(np.random.randint(180, 255))
        val = int(np.random.randint(180, 255))
        hsv_pixel = np.uint8([[[hue, sat, val]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        palette.append((int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])))
    return palette


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
    raise FileNotFoundError(f"Could not find img1 directory for {seq_name} in {data_dir}")


def load_tracks(track_file: Path) -> Dict[int, List[Tuple[int, np.ndarray, float]]]:
    tracks_by_frame: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}
    if not track_file.is_file():
        return tracks_by_frame
    with track_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                parts = line.split()
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            trk_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            box = np.array([x, y, x + w, y + h], dtype=float)
            tracks_by_frame.setdefault(frame, []).append((trk_id, box, conf))
    return tracks_by_frame


def load_id_switches(
    switches_csv: Path, seq_name: str, tracker_name: str
) -> Dict[int, List[Dict[str, Any]]]:
    switches_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    if not switches_csv.is_file():
        return switches_by_frame

    with switches_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sequence") == seq_name and row.get("tracker") == tracker_name:
                frame = int(row["frame"])
                switches_by_frame.setdefault(frame, []).append({
                    "gt_id": int(row["gt_id"]),
                    "old_pred_id": int(row["old_pred_id"]),
                    "new_pred_id": int(row["new_pred_id"]),
                    "causes": row.get("causes", ""),
                })
    return switches_by_frame


def render_panel(
    frame_img: np.ndarray,
    frame_idx: int,
    tracks: List[Tuple[int, np.ndarray, float]],
    trail_history: Dict[int, deque],
    recent_switches: List[Tuple[int, int, Tuple[int, int]]],  # (start_frame, trk_id, foot_pos)
    palette: List[Tuple[int, int, int]],
    trail_len: int = 40,
    ring_duration: int = 15,
) -> np.ndarray:
    canvas = frame_img.copy()

    # Update trails
    current_ids = set()
    for trk_id, box, conf in tracks:
        current_ids.add(trk_id)
        foot_x = int(round((box[0] + box[2]) / 2.0))
        foot_y = int(round(box[3]))
        if trk_id not in trail_history:
            trail_history[trk_id] = deque(maxlen=trail_len)
        trail_history[trk_id].append((foot_x, foot_y))

    # Clean dead trails
    dead_ids = [tid for tid in trail_history if tid not in current_ids and len(trail_history[tid]) == 0]
    for tid in dead_ids:
        del trail_history[tid]

    # Draw trails
    for trk_id, pts in trail_history.items():
        color = palette[trk_id % len(palette)]
        num_pts = len(pts)
        if num_pts < 2:
            continue
        for i in range(num_pts - 1):
            pt1 = pts[i]
            pt2 = pts[i + 1]
            alpha = (i + 1) / float(num_pts)
            # Fade towards dark / transparent
            faded_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
            thickness = max(1, int(round(1 + 2 * alpha)))
            cv2.line(canvas, pt1, pt2, faded_color, thickness, cv2.LINE_AA)

    # Draw bounding boxes and labels
    for trk_id, box, conf in tracks:
        color = palette[trk_id % len(palette)]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label background
        label_text = f"ID:{trk_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        lbl_y1 = max(0, y1 - txt_h - 4)
        cv2.rectangle(canvas, (x1, lbl_y1), (x1 + txt_w + 4, lbl_y1 + txt_h + 4), color, -1)
        cv2.putText(
            canvas,
            label_text,
            (x1 + 2, lbl_y1 + txt_h + 1),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # Draw ring markers for recent ID switches
    for sw_start, sw_id, (sw_x, sw_y) in recent_switches:
        elapsed = frame_idx - sw_start
        if 0 <= elapsed <= ring_duration:
            radius = int(12 + elapsed * 2.5)
            alpha = 1.0 - (elapsed / float(ring_duration))
            ring_color = (0, 0, 255)  # Red marker
            cv2.circle(canvas, (sw_x, sw_y), radius, ring_color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "ID SW",
                (sw_x - 18, sw_y - radius - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    return canvas


def render_comparison_video(
    seq_name: str,
    data_dir: Path,
    tracks_dir: Path,
    switches_csv: Path,
    output_path: Path,
    num_frames: int = 300,
    fps: float = 30.0,
    trail_len: int = 40,
    scale: float = 0.5,
) -> None:
    img_dir = find_image_dir(seq_name, data_dir)
    sort_tracks = load_tracks(tracks_dir / "sort" / f"{seq_name}.txt")
    deepsort_tracks = load_tracks(tracks_dir / "deepsort" / f"{seq_name}.txt")

    sort_switches_by_frame = load_id_switches(switches_csv, seq_name, "sort")
    deepsort_switches_by_frame = load_id_switches(switches_csv, seq_name, "deepsort")

    palette = generate_color_palette(128)

    sort_trails: Dict[int, deque] = {}
    deepsort_trails: Dict[int, deque] = {}

    recent_switches_sort: List[Tuple[int, int, Tuple[int, int]]] = []
    recent_switches_deepsort: List[Tuple[int, int, Tuple[int, int]]] = []

    # Read frame 1 to get dimensions
    sample_img = cv2.imread(str(img_dir / "000001.jpg"))
    if sample_img is None:
        raise FileNotFoundError(f"Could not load first frame 000001.jpg from {img_dir}")

    h_orig, w_orig = sample_img.shape[:2]
    w_scaled = int(round(w_orig * scale))
    h_scaled = int(round(h_orig * scale))

    banner_h = 42
    total_w = w_scaled * 2
    total_h = h_scaled + banner_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (total_w, total_h))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    print(f"Rendering {num_frames} frames of {seq_name} (SORT vs DeepSORT) to {output_path}...")

    for f_idx in range(1, num_frames + 1):
        img_path = img_dir / f"{f_idx:06d}.jpg"
        if not img_path.is_file():
            break

        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            break

        s_trks = sort_tracks.get(f_idx, [])
        ds_trks = deepsort_tracks.get(f_idx, [])

        # Check for new switches in SORT
        for sw in sort_switches_by_frame.get(f_idx, []):
            new_id = sw["new_pred_id"]
            # find foot point
            for tid, box, _ in s_trks:
                if tid == new_id:
                    foot = (int((box[0] + box[2]) / 2), int(box[3]))
                    recent_switches_sort.append((f_idx, tid, foot))
                    break

        # Check for new switches in DeepSORT
        for sw in deepsort_switches_by_frame.get(f_idx, []):
            new_id = sw["new_pred_id"]
            for tid, box, _ in ds_trks:
                if tid == new_id:
                    foot = (int((box[0] + box[2]) / 2), int(box[3]))
                    recent_switches_deepsort.append((f_idx, tid, foot))
                    break

        # Filter out switches older than 20 frames
        recent_switches_sort = [s for s in recent_switches_sort if f_idx - s[0] <= 20]
        recent_switches_deepsort = [s for s in recent_switches_deepsort if f_idx - s[0] <= 20]

        # Render individual panels
        panel_sort = render_panel(
            frame_bgr,
            f_idx,
            s_trks,
            sort_trails,
            recent_switches_sort,
            palette,
            trail_len=trail_len,
        )
        panel_ds = render_panel(
            frame_bgr,
            f_idx,
            ds_trks,
            deepsort_trails,
            recent_switches_deepsort,
            palette,
            trail_len=trail_len,
        )

        # Scale down
        p_sort_small = cv2.resize(panel_sort, (w_scaled, h_scaled))
        p_ds_small = cv2.resize(panel_ds, (w_scaled, h_scaled))

        # Build composite with top header
        banner = np.full((banner_h, total_w, 3), 30, dtype=np.uint8)

        # Left header
        sort_hdr = f"SORT | Frame: {f_idx:03d}/{num_frames} | Tracks: {len(s_trks)}"
        cv2.putText(banner, sort_hdr, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Divider in banner
        cv2.line(banner, (w_scaled, 6), (w_scaled, banner_h - 6), (100, 100, 100), 2)

        # Right header
        ds_hdr = f"DeepSORT | Frame: {f_idx:03d}/{num_frames} | Tracks: {len(ds_trks)}"
        cv2.putText(banner, ds_hdr, (w_scaled + 15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        body = np.hstack([p_sort_small, p_ds_small])
        full_frame = np.vstack([banner, body])

        writer.write(full_frame)

        if f_idx % 50 == 0 or f_idx == num_frames:
            print(f"  Frame {f_idx}/{num_frames} rendered...")

    writer.release()
    print(f"Video saved to {output_path} ({total_w}x{total_h} at {fps} FPS)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render side-by-side comparison video of SORT vs DeepSORT."
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="MOT17-04",
        help="Sequence name to render",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Path to MOT sequences directory",
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("results/tracks"),
        help="Path to tracker results directory",
    )
    parser.add_argument(
        "--switches-csv",
        type=Path,
        default=Path("results/id_switches.csv"),
        help="Path to ID switches CSV",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("results/figures/clip_sort_vs_deepsort.mp4"),
        help="Path to output video file",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=300,
        help="Number of frames to render",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback frame rate",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=40,
        help="Length of trajectory trail in frames",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scaling factor for each video panel",
    )

    args = parser.parse_args(argv)

    render_comparison_video(
        seq_name=args.seq,
        data_dir=args.data_dir,
        tracks_dir=args.tracks_dir,
        switches_csv=args.switches_csv,
        output_path=args.output_video,
        num_frames=args.num_frames,
        fps=args.fps,
        trail_len=args.trail_length,
        scale=args.scale,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
