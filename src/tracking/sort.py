from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.kalman import KalmanFilter


def xyxy_to_cxcyah(bbox: np.ndarray) -> np.ndarray:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    a = w / np.maximum(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=float)


def cxcyah_to_xyxy(cxcyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = cxcyah[:4]
    w = a * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=float)


def xyxy_to_tlwh(bbox: np.ndarray) -> np.ndarray:
    return np.array(
        [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=float
    )


def tlwh_to_xyxy(tlwh: np.ndarray) -> np.ndarray:
    return np.array(
        [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], dtype=float
    )


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.empty((boxes_a.shape[0], boxes_b.shape[0]), dtype=float)

    boxes_a = np.asarray(boxes_a, dtype=float)
    boxes_b = np.asarray(boxes_b, dtype=float)

    x1 = np.maximum(boxes_a[:, 0, None], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, 1, None], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, 3, None], boxes_b[None, :, 3])

    intersection_w = np.maximum(0.0, x2 - x1)
    intersection_h = np.maximum(0.0, y2 - y1)
    intersection = intersection_w * intersection_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(union, 1e-12)

    return intersection / union


def linear_assignment(
    cost_matrix: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: List[Tuple[int, int]] = []
    unmatched_a: List[int] = []
    unmatched_b: List[int] = []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= threshold:
            matches.append((r, c))

    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}

    for r in range(cost_matrix.shape[0]):
        if r not in matched_rows:
            unmatched_a.append(r)

    for c in range(cost_matrix.shape[1]):
        if c not in matched_cols:
            unmatched_b.append(c)

    matches_arr = (
        np.array(matches, dtype=int)
        if len(matches) > 0
        else np.empty((0, 2), dtype=int)
    )
    return matches_arr, np.array(unmatched_a, dtype=int), np.array(unmatched_b, dtype=int)


class KalmanBoxTracker:
    count: int = 0

    def __init__(self, bbox: np.ndarray, score: float = 1.0) -> None:
        self.kf = KalmanFilter()
        measurement = xyxy_to_cxcyah(bbox)
        self.mean, self.covariance = self.kf.initiate(measurement)

        self.time_since_update: int = 0
        KalmanBoxTracker.count += 1
        self.id: int = KalmanBoxTracker.count
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 0
        self.score: float = float(score)

    def update(self, bbox: np.ndarray, score: float = 1.0) -> None:
        measurement = xyxy_to_cxcyah(bbox)
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, measurement
        )
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)

    def predict(self) -> np.ndarray:
        if self.mean[3] + self.mean[7] <= 0:
            # Prevent negative height projection
            self.mean[7] = 0.0

        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return cxcyah_to_xyxy(self.mean[:4])


class Sort:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    @classmethod
    def reset_id_counter(cls) -> None:
        KalmanBoxTracker.count = 0

    def update(self, dets: np.ndarray = np.empty((0, 5))) -> np.ndarray:
        self.frame_count += 1

        if dets is None or len(dets) == 0:
            dets = np.empty((0, 5), dtype=float)
        elif dets.shape[1] == 4:
            # If scores not provided, append dummy score 1.0
            scores = np.ones((dets.shape[0], 1), dtype=float)
            dets = np.hstack([dets, scores])

        # Get predicted locations from existing trackers
        predicted_boxes = np.empty((len(self.trackers), 4), dtype=float)
        to_del: List[int] = []

        for idx, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(idx)
            else:
                predicted_boxes[idx] = pos

        for idx in reversed(to_del):
            self.trackers.pop(idx)
        if to_del:
            predicted_boxes = np.delete(predicted_boxes, to_del, axis=0)

        # Compute IoU cost matrix
        if len(self.trackers) > 0 and len(dets) > 0:
            iou_mat = iou_batch(predicted_boxes, dets[:, :4])
            cost_matrix = 1.0 - iou_mat
            matches, unmatched_trks, unmatched_dets = linear_assignment(
                cost_matrix, threshold=1.0 - self.iou_threshold
            )
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_trks = np.arange(len(self.trackers), dtype=int)
            unmatched_dets = np.arange(len(dets), dtype=int)

        # Update matched trackers
        for trk_idx, det_idx in matches:
            self.trackers[trk_idx].update(dets[det_idx, :4], score=dets[det_idx, 4])

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(dets[det_idx, :4], score=dets[det_idx, 4])
            self.trackers.append(new_trk)

        active_outputs: List[np.ndarray] = []
        surviving_trackers: List[KalmanBoxTracker] = []

        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                continue

            surviving_trackers.append(trk)

            # Output tracks that were updated this frame and have sufficient hits
            if trk.time_since_update == 0 and (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ):
                box = trk.get_state()
                active_outputs.append(
                    np.array([box[0], box[1], box[2], box[3], trk.id, trk.score])
                )

        self.trackers = surviving_trackers

        if active_outputs:
            return np.array(active_outputs, dtype=float)
        return np.empty((0, 6), dtype=float)
