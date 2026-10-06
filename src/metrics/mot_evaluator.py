from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data.mot_loader import GroundTruth, MOTSequence, load_sequence
from src.tracker.matching import bbox_ious


@dataclass
class FramePrediction:
    frame: int
    track_id: int
    bbox_tlwh: np.ndarray
    score: float

    @property
    def bbox_tlbr(self) -> np.ndarray:
        x, y, w, h = self.bbox_tlwh
        return np.array([x, y, x + w, y + h], dtype=float)


@dataclass
class MOTMetrics:
    num_frames: int
    num_objects: int
    num_predictions: int
    mota: float
    motp: float
    idf1: float
    idp: float
    idr: float
    recall: float
    precision: float
    num_false_positives: int
    num_misses: int
    num_switches: int
    mostly_tracked: int
    mostly_lost: int
    partially_tracked: int
    num_unique_gt_ids: int
    num_unique_pred_ids: int
    fps: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def load_predictions(pred_path: Path) -> Dict[int, List[FramePrediction]]:
    predictions_by_frame: Dict[int, List[FramePrediction]] = {}
    if not pred_path.exists():
        return predictions_by_frame

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                parts = line.split()
            if len(parts) < 6:
                continue

            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6]) if len(parts) > 6 else 1.0

            predictions_by_frame.setdefault(frame, []).append(
                FramePrediction(
                    frame=frame,
                    track_id=track_id,
                    bbox_tlwh=np.array([x, y, w, h], dtype=float),
                    score=score,
                )
            )
    return predictions_by_frame


def evaluate_sequence(
    sequence: MOTSequence,
    predictions_by_frame: Dict[int, List[FramePrediction]],
    iou_threshold: float = 0.5,
    min_visibility: float = 0.0,
    elapsed_time_sec: float = 0.0,
) -> MOTMetrics:
    gt_by_frame = sequence.get_ground_truths_by_frame(
        pedestrian_only=True, min_visibility=min_visibility
    )
    all_frames = sorted(set(list(gt_by_frame.keys()) + list(predictions_by_frame.keys())))

    # Non-pedestrian or ignore annotations used to prevent penalizing predictions in non-target regions
    ignore_by_frame: Dict[int, List[GroundTruth]] = {}
    for gt in sequence.ground_truths:
        if gt.class_id != 1 or gt.visibility < min_visibility or gt.conf == 0:
            ignore_by_frame.setdefault(gt.frame, []).append(gt)

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_iou_sum = 0.0

    last_assigned_pred: Dict[int, int] = {}
    gt_frame_counts: Dict[int, int] = {}
    gt_tp_counts: Dict[int, int] = {}

    overlap_counts: Dict[Tuple[int, int], int] = {}
    gt_lengths: Dict[int, int] = {}
    pred_lengths: Dict[int, int] = {}

    for frame in all_frames:
        gts = gt_by_frame.get(frame, [])
        preds = predictions_by_frame.get(frame, [])
        ignores = ignore_by_frame.get(frame, [])

        total_gt += len(gts)
        total_pred += len(preds)

        for gt in gts:
            gt_frame_counts[gt.track_id] = gt_frame_counts.get(gt.track_id, 0) + 1
            gt_lengths[gt.track_id] = gt_lengths.get(gt.track_id, 0) + 1

        for pred in preds:
            pred_lengths[pred.track_id] = pred_lengths.get(pred.track_id, 0) + 1

        if len(gts) == 0 and len(preds) == 0:
            continue

        if len(gts) == 0:
            for pred in preds:
                # Discard false alarm if hypothesis overlaps an ignore region
                if len(ignores) > 0:
                    ig_boxes = np.array([ig.bbox_tlbr for ig in ignores])
                    p_box = np.array([pred.bbox_tlbr])
                    ious = bbox_ious(p_box, ig_boxes)
                    if np.max(ious) >= iou_threshold:
                        continue
                total_fp += 1
            continue

        if len(preds) == 0:
            total_fn += len(gts)
            continue

        gt_boxes = np.array([gt.bbox_tlbr for gt in gts])
        pred_boxes = np.array([pred.bbox_tlbr for preds_i in preds for pred in [preds_i]])
        ious = bbox_ious(gt_boxes, pred_boxes)

        cost_matrix = 1.0 - ious
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_gt_indices: Set[int] = set()
        matched_pred_indices: Set[int] = set()

        for r, c in zip(row_ind, col_ind):
            iou = ious[r, c]
            if iou >= iou_threshold:
                matched_gt_indices.add(r)
                matched_pred_indices.add(c)
                total_tp += 1
                total_iou_sum += float(iou)

                gt_id = gts[r].track_id
                pred_id = preds[c].track_id

                gt_tp_counts[gt_id] = gt_tp_counts.get(gt_id, 0) + 1
                pair = (gt_id, pred_id)
                overlap_counts[pair] = overlap_counts.get(pair, 0) + 1

                if gt_id in last_assigned_pred:
                    if last_assigned_pred[gt_id] != pred_id:
                        total_id_switches += 1
                last_assigned_pred[gt_id] = pred_id

        # Compute misses
        for r, gt in enumerate(gts):
            if r not in matched_gt_indices:
                total_fn += 1

        # Compute false positives after ignore check
        for c, pred in enumerate(preds):
            if c not in matched_pred_indices:
                if len(ignores) > 0:
                    ig_boxes = np.array([ig.bbox_tlbr for ig in ignores])
                    p_box = np.array([pred.bbox_tlbr])
                    ig_ious = bbox_ious(p_box, ig_boxes)
                    if np.max(ig_ious) >= iou_threshold:
                        continue
                total_fp += 1

    # CLEAR MOT metrics
    mota = (
        (1.0 - (total_fn + total_fp + total_id_switches) / total_gt)
        if total_gt > 0
        else 0.0
    )
    motp = (total_iou_sum / total_tp) if total_tp > 0 else 0.0
    recall = (total_tp / total_gt) if total_gt > 0 else 0.0
    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0

    # Mostly Tracked (MT), Mostly Lost (ML), Partially Tracked (PT)
    mt = 0
    ml = 0
    pt = 0
    for gt_id, count in gt_frame_counts.items():
        tp_count = gt_tp_counts.get(gt_id, 0)
        ratio = tp_count / count
        if ratio >= 0.8:
            mt += 1
        elif ratio <= 0.2:
            ml += 1
        else:
            pt += 1

    # Global trajectory matching for IDF1
    all_gt_ids = list(gt_lengths.keys())
    all_pred_ids = list(pred_lengths.keys())
    gt_id_to_idx = {gid: idx for idx, gid in enumerate(all_gt_ids)}
    pred_id_to_idx = {pid: idx for idx, pid in enumerate(all_pred_ids)}

    if len(all_gt_ids) > 0 and len(all_pred_ids) > 0:
        cost_matrix_id = np.zeros((len(all_gt_ids), len(all_pred_ids)), dtype=float)
        for (gid, pid), cnt in overlap_counts.items():
            if gid in gt_id_to_idx and pid in pred_id_to_idx:
                cost_matrix_id[gt_id_to_idx[gid], pred_id_to_idx[pid]] = -cnt

        row_ind_id, col_ind_id = linear_sum_assignment(cost_matrix_id)
        idtp = 0
        for r, c in zip(row_ind_id, col_ind_id):
            idtp += int(-cost_matrix_id[r, c])
    else:
        idtp = 0

    idfp = sum(pred_lengths.values()) - idtp
    idfn = sum(gt_lengths.values()) - idtp
    idf1 = (2.0 * idtp / (2.0 * idtp + idfp + idfn)) if (2.0 * idtp + idfp + idfn) > 0 else 0.0
    idp = (idtp / (idtp + idfp)) if (idtp + idfp) > 0 else 0.0
    idr = (idtp / (idtp + idfn)) if (idtp + idfn) > 0 else 0.0

    num_frames = len(all_frames)
    fps = (num_frames / elapsed_time_sec) if elapsed_time_sec > 0.0 else 0.0

    return MOTMetrics(
        num_frames=num_frames,
        num_objects=total_gt,
        num_predictions=total_pred,
        mota=float(mota),
        motp=float(motp),
        idf1=float(idf1),
        idp=float(idp),
        idr=float(idr),
        recall=float(recall),
        precision=float(precision),
        num_false_positives=total_fp,
        num_misses=total_fn,
        num_switches=total_id_switches,
        mostly_tracked=mt,
        mostly_lost=ml,
        partially_tracked=pt,
        num_unique_gt_ids=len(all_gt_ids),
        num_unique_pred_ids=len(all_pred_ids),
        fps=float(fps),
    )
