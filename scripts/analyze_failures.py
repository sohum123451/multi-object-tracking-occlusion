import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tracking.embed import OSNetExtractor
from src.tracking.sort import iou_batch, linear_assignment


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


def find_gt_file(seq_name: str, data_dir: Path) -> Path:
    direct = data_dir / seq_name / "gt" / "gt.txt"
    if direct.is_file():
        return direct
    frcnn = data_dir / f"{seq_name}-FRCNN" / "gt" / "gt.txt"
    if frcnn.is_file():
        return frcnn
    matches = list(data_dir.glob(f"{seq_name}*/gt/gt.txt"))
    if matches and matches[0].is_file():
        return matches[0]
    raise FileNotFoundError(f"Could not find gt.txt for {seq_name} in {data_dir}")


def load_gt_pedestrians(gt_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    gt_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            # Filter pedestrian class (class 1)
            if parts[7] != "1":
                continue
            frame = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            vis = float(parts[8])
            box = np.array([x, y, x + w, y + h], dtype=float)
            gt_by_frame.setdefault(frame, []).append({
                "id": gt_id,
                "box": box,
                "vis": vis,
            })
    return gt_by_frame


def load_tracker_predictions(track_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    pred_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    if not track_path.is_file():
        return pred_by_frame
    with track_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                parts = line.split()
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            pred_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            box = np.array([x, y, x + w, y + h], dtype=float)
            pred_by_frame.setdefault(frame, []).append({
                "id": pred_id,
                "box": box,
            })
    return pred_by_frame


def compute_gt_mean_embeddings(
    seq_name: str,
    gt_by_frame: Dict[int, List[Dict[str, Any]]],
    img_dir: Path,
    extractor: OSNetExtractor,
    max_crops_per_id: int = 5,
) -> Dict[int, np.ndarray]:
    crops_by_id: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for frame, items in sorted(gt_by_frame.items()):
        if frame % 10 != 1:
            continue
        for it in items:
            gid = it["id"]
            if len(crops_by_id.get(gid, [])) < max_crops_per_id:
                crops_by_id.setdefault(gid, []).append((frame, it["box"]))

    all_crops: List[np.ndarray] = []
    crop_owners: List[int] = []

    for gid, crop_items in crops_by_id.items():
        for frame, box in crop_items:
            img_path = img_dir / f"{frame:06d}.jpg"
            if not img_path.is_file():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            x1 = max(0, int(round(box[0])))
            y1 = max(0, int(round(box[1])))
            x2 = min(w_img, int(round(box[2])))
            y2 = min(h_img, int(round(box[3])))
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                all_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crop_owners.append(gid)

    if not all_crops:
        return {}

    feats = extractor.extract(all_crops)
    mean_embeddings: Dict[int, np.ndarray] = {}
    owner_arr = np.array(crop_owners)

    for gid in set(crop_owners):
        idx = np.where(owner_arr == gid)[0]
        id_feats = feats[idx]
        mean_v = np.mean(id_feats, axis=0)
        norm = np.linalg.norm(mean_v)
        if norm > 1e-12:
            mean_v = mean_v / norm
        mean_embeddings[gid] = mean_v.astype(np.float32)

    return mean_embeddings


def analyze_sequence_id_switches(
    seq_name: str,
    tracker_name: str,
    gt_by_frame: Dict[int, List[Dict[str, Any]]],
    pred_by_frame: Dict[int, List[Dict[str, Any]]],
    gt_embeddings: Dict[int, np.ndarray],
    vis_thresh: float,
    occ_iou_thresh: float,
    speed_thresh: float,
    app_thresh: float,
    match_iou_thresh: float = 0.5,
) -> List[Dict[str, Any]]:
    # Map each GT id to its trajectory of centers: {gt_id: {frame: (cx, cy)}}
    gt_centers: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for frame, items in gt_by_frame.items():
        for it in items:
            b = it["box"]
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            gt_centers.setdefault(it["id"], {})[frame] = (cx, cy)

    # State tracking: {gt_id: (last_matched_pred_id, last_matched_frame, last_box)}
    gt_last_matched: Dict[int, Tuple[int, int, np.ndarray]] = {}
    # Reverse tracking: {pred_id: (last_matched_gt_id, last_matched_frame)}
    pred_last_gt: Dict[int, Tuple[int, int]] = {}

    max_frame = max(
        max(gt_by_frame.keys(), default=0),
        max(pred_by_frame.keys(), default=0),
    )

    id_switches: List[Dict[str, Any]] = []

    for frame in range(1, max_frame + 1):
        gts = gt_by_frame.get(frame, [])
        preds = pred_by_frame.get(frame, [])
        if not gts or not preds:
            continue

        gt_boxes = np.array([g["box"] for g in gts])
        pred_boxes = np.array([p["box"] for p in preds])

        ious = iou_batch(gt_boxes, pred_boxes)
        cost_mat = 1.0 - ious
        matches, _, _ = linear_assignment(cost_mat, threshold=1.0 - match_iou_thresh)

        # Mapping for current frame
        curr_gt_to_pred: Dict[int, int] = {}
        curr_pred_to_gt: Dict[int, int] = {}

        for g_idx, p_idx in matches:
            gid = gts[g_idx]["id"]
            pid = preds[p_idx]["id"]
            curr_gt_to_pred[gid] = pid
            curr_pred_to_gt[pid] = gid

            if gid in gt_last_matched:
                old_pid, prev_frame, prev_box = gt_last_matched[gid]
                if pid != old_pid:
                    # ID switch detected
                    vis = gts[g_idx]["vis"]

                    # Compute max IoU with other GT boxes in this frame
                    other_gt_boxes = [g["box"] for j, g in enumerate(gts) if j != g_idx]
                    if other_gt_boxes:
                        other_gt_arr = np.array(other_gt_boxes)
                        inter_ious = iou_batch(gt_boxes[g_idx : g_idx + 1], other_gt_arr)[0]
                        max_gt_iou = float(np.max(inter_ious))
                    else:
                        max_gt_iou = 0.0

                    # Compute speed in px/frame
                    curr_cx = (gt_boxes[g_idx][0] + gt_boxes[g_idx][2]) / 2.0
                    curr_cy = (gt_boxes[g_idx][1] + gt_boxes[g_idx][3]) / 2.0
                    if prev_frame in gt_centers[gid]:
                        prev_cx, prev_cy = gt_centers[gid][prev_frame]
                        delta_frames = max(1, frame - prev_frame)
                        dist = np.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
                        speed = float(dist / delta_frames)
                    else:
                        speed = 0.0

                    # Identify the other GT identity involved
                    other_gid: Optional[int] = None
                    if pid in pred_last_gt:
                        cand_gid, _ = pred_last_gt[pid]
                        if cand_gid != gid:
                            other_gid = cand_gid
                    if other_gid is None:
                        # Find other GT with highest spatial overlap in this frame
                        if other_gt_boxes:
                            other_gids = [g["id"] for j, g in enumerate(gts) if j != g_idx]
                            max_iou_idx = int(np.argmax(inter_ious))
                            if inter_ious[max_iou_idx] > 0.05:
                                other_gid = other_gids[max_iou_idx]

                    # Compute cosine similarity between mean embeddings
                    cos_sim = 0.0
                    if other_gid is not None and gid in gt_embeddings and other_gid in gt_embeddings:
                        cos_sim = float(np.dot(gt_embeddings[gid], gt_embeddings[other_gid]))

                    # Classify cause
                    causes: List[str] = []
                    if vis < vis_thresh or max_gt_iou >= occ_iou_thresh:
                        causes.append("occlusion")
                    if speed >= speed_thresh:
                        causes.append("fast_motion")
                    if cos_sim >= app_thresh:
                        causes.append("similar_appearance")
                    if not causes:
                        causes.append("other")

                    severity = (1.0 - vis) + 2.0 * max_gt_iou + (speed / 20.0) + max(0.0, cos_sim)

                    id_switches.append({
                        "tracker": tracker_name,
                        "sequence": seq_name,
                        "frame": frame,
                        "prev_frame": prev_frame,
                        "gt_id": gid,
                        "other_gt_id": other_gid if other_gid is not None else -1,
                        "old_pred_id": old_pid,
                        "new_pred_id": pid,
                        "gt_visibility": round(vis, 4),
                        "max_gt_iou": round(max_gt_iou, 4),
                        "gt_speed": round(speed, 2),
                        "cos_sim": round(cos_sim, 4),
                        "causes": ",".join(causes),
                        "severity": severity,
                        "gt_box": gt_boxes[g_idx],
                        "prev_gt_box": prev_box,
                    })

            gt_last_matched[gid] = (pid, frame, gt_boxes[g_idx])
            pred_last_gt[pid] = (gid, frame)

    return id_switches


def save_worst_switch_crops(
    switches: List[Dict[str, Any]],
    data_dir: Path,
    output_dir: Path,
    top_k: int = 10,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_switches = sorted(switches, key=lambda s: s["severity"], reverse=True)[:top_k]

    for rank, sw in enumerate(sorted_switches, start=1):
        seq = sw["sequence"]
        frame = sw["frame"]
        prev_frame = sw["prev_frame"]
        img_dir = find_image_dir(seq, data_dir)

        img_curr = cv2.imread(str(img_dir / f"{frame:06d}.jpg"))
        img_prev = cv2.imread(str(img_dir / f"{prev_frame:06d}.jpg"))

        if img_curr is None or img_prev is None:
            continue

        def get_padded_crop(img: np.ndarray, box: np.ndarray) -> np.ndarray:
            h_img, w_img = img.shape[:2]
            pad_x = (box[2] - box[0]) * 0.15
            pad_y = (box[3] - box[1]) * 0.15
            x1 = max(0, int(box[0] - pad_x))
            y1 = max(0, int(box[1] - pad_y))
            x2 = min(w_img, int(box[2] + pad_x))
            y2 = min(h_img, int(box[3] + pad_y))
            crop = img[y1:y2, x1:x2].copy()
            # Draw local box on crop
            bx1 = int(box[0] - x1)
            by1 = int(box[1] - y1)
            bx2 = int(box[2] - x1)
            by2 = int(box[3] - y1)
            cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            return crop

        crop_prev = get_padded_crop(img_prev, sw["prev_gt_box"])
        crop_curr = get_padded_crop(img_curr, sw["gt_box"])

        target_h = 240
        def resize_to_h(c: np.ndarray, h: int) -> np.ndarray:
            scale = h / max(1, c.shape[0])
            w = max(1, int(round(c.shape[1] * scale)))
            return cv2.resize(c, (w, h))

        c1 = resize_to_h(crop_prev, target_h)
        c2 = resize_to_h(crop_curr, target_h)

        divider = np.full((target_h, 4, 3), 200, dtype=np.uint8)
        combined = np.hstack([c1, divider, c2])

        # Header banner
        header = np.zeros((45, combined.shape[1], 3), dtype=np.uint8)
        title = f"Rank {rank}: {sw['tracker'].upper()} {seq} GT:{sw['gt_id']} P:{sw['old_pred_id']}->{sw['new_pred_id']}"
        subtitle = f"F{prev_frame}->F{frame} | Vis:{sw['gt_visibility']} | IoU:{sw['max_gt_iou']} | Spd:{sw['gt_speed']} | {sw['causes']}"
        cv2.putText(header, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(header, subtitle, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1, cv2.LINE_AA)

        final_composite = np.vstack([header, combined])
        out_file = output_dir / f"worst_switch_{rank:02d}_{sw['tracker']}_{seq}_f{frame}.jpg"
        cv2.imwrite(str(out_file), final_composite)


def plot_cause_bar_chart(
    switches: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    causes_list = ["occlusion", "fast_motion", "similar_appearance", "other"]

    counts: Dict[str, Dict[str, int]] = {
        "sort": {c: 0 for c in causes_list},
        "deepsort": {c: 0 for c in causes_list},
    }

    for sw in switches:
        trk = sw["tracker"]
        c_list = [c.strip() for c in sw["causes"].split(",")]
        for c in c_list:
            if c in counts[trk]:
                counts[trk][c] += 1

    x = np.arange(len(causes_list))
    width = 0.35

    sort_vals = [counts["sort"][c] for c in causes_list]
    deepsort_vals = [counts["deepsort"][c] for c in causes_list]

    labels = ["Occlusion", "Fast Motion", "Similar Appearance", "Other"]

    plt.figure(figsize=(9, 5.5))
    rects1 = plt.bar(x - width / 2, sort_vals, width, label="SORT", color="#4e79a7")
    rects2 = plt.bar(x + width / 2, deepsort_vals, width, label="DeepSORT", color="#e15759")

    plt.ylabel("Number of ID Switches")
    plt.title("ID Switch Failure Cause Breakdown (MOT17 Train Subset)")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects: Any) -> None:
        for rect in rects:
            height = rect.get_height()
            plt.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze tracking failures and classify ID switch causes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/MOT17/train"),
        help="Path to MOT ground truth sequences",
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("results/tracks"),
        help="Path to tracker output files",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["sort", "deepsort"],
        help="Trackers to analyze",
    )
    parser.add_argument(
        "--seq",
        nargs="+",
        default=["MOT17-02", "MOT17-04", "MOT17-09"],
        help="Sequences to analyze",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.6,
        help="Visibility threshold below which an ID switch is classified as occlusion",
    )
    parser.add_argument(
        "--occlusion-iou-threshold",
        type=float,
        default=0.2,
        help="Inter-GT IoU threshold above which an ID switch is classified as occlusion",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=15.0,
        help="Target speed threshold (px/frame) to classify fast motion",
    )
    parser.add_argument(
        "--appearance-threshold",
        type=float,
        default=0.70,
        help="Cosine similarity threshold between GT embeddings to classify similar appearance",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/id_switches.csv"),
        help="Path to save ID switch catalog CSV",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to save figures and crop pairs",
    )

    args = parser.parse_args(argv)

    extractor = OSNetExtractor()
    all_switches: List[Dict[str, Any]] = []

    for seq in args.seq:
        gt_file = find_gt_file(seq, args.data_dir)
        img_dir = find_image_dir(seq, args.data_dir)
        gt_by_frame = load_gt_pedestrians(gt_file)

        print(f"Extracting mean GT embeddings for {seq}...")
        gt_embeddings = compute_gt_mean_embeddings(
            seq_name=seq,
            gt_by_frame=gt_by_frame,
            img_dir=img_dir,
            extractor=extractor,
        )

        for tracker in args.trackers:
            track_file = args.tracks_dir / tracker / f"{seq}.txt"
            pred_by_frame = load_tracker_predictions(track_file)

            switches = analyze_sequence_id_switches(
                seq_name=seq,
                tracker_name=tracker,
                gt_by_frame=gt_by_frame,
                pred_by_frame=pred_by_frame,
                gt_embeddings=gt_embeddings,
                vis_thresh=args.visibility_threshold,
                occ_iou_thresh=args.occlusion_iou_threshold,
                speed_thresh=args.speed_threshold,
                app_thresh=args.appearance_threshold,
            )
            all_switches.extend(switches)
            print(f"[{tracker.upper()}] {seq}: found {len(switches)} ID switches")

    # Write results/id_switches.csv
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tracker",
        "sequence",
        "frame",
        "gt_id",
        "other_gt_id",
        "old_pred_id",
        "new_pred_id",
        "gt_visibility",
        "max_gt_iou",
        "gt_speed",
        "cos_sim",
        "causes",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for sw in all_switches:
            writer.writerow(sw)

    print(f"\nID switches catalog written to {args.output_csv} ({len(all_switches)} total switches)")

    # Print summary table per tracker and cause
    causes_list = ["occlusion", "fast_motion", "similar_appearance", "other"]
    summary_counts: Dict[str, Dict[str, int]] = {
        trk: {c: 0 for c in causes_list} for trk in args.trackers
    }
    total_counts: Dict[str, int] = {trk: 0 for trk in args.trackers}

    for sw in all_switches:
        trk = sw["tracker"]
        total_counts[trk] += 1
        for c in [c.strip() for c in sw["causes"].split(",")]:
            if c in summary_counts[trk]:
                summary_counts[trk][c] += 1

    print("\nID Switch Causes Summary Table:")
    headers = ["Tracker", "Total Switches", "Occlusion", "Fast Motion", "Similar App", "Other"]
    row_fmt = "{:<12} | {:<14} | {:<10} | {:<12} | {:<12} | {:<8}"
    print(row_fmt.format(*headers))
    print("-" * 80)
    for trk in args.trackers:
        tot = total_counts[trk]
        occ = summary_counts[trk]["occlusion"]
        spd = summary_counts[trk]["fast_motion"]
        app = summary_counts[trk]["similar_appearance"]
        oth = summary_counts[trk]["other"]
        print(row_fmt.format(
            trk,
            str(tot),
            f"{occ} ({occ / tot * 100:.1f}%)" if tot > 0 else "0",
            f"{spd} ({spd / tot * 100:.1f}%)" if tot > 0 else "0",
            f"{app} ({app / tot * 100:.1f}%)" if tot > 0 else "0",
            f"{oth} ({oth / tot * 100:.1f}%)" if tot > 0 else "0",
        ))
    print("-" * 80)

    # Plot bar chart
    chart_path = args.figures_dir / "id_switch_causes.png"
    plot_cause_bar_chart(all_switches, chart_path)
    print(f"Cause breakdown bar chart saved to {chart_path}")

    # Save cropped image pairs for top 10 worst switches
    crops_dir = args.figures_dir / "worst_switches"
    save_worst_switch_crops(all_switches, args.data_dir, crops_dir, top_k=10)
    print(f"Top 10 worst switch crop pairs saved to {crops_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
