from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

from src.tracking.kalman import KalmanFilter
from src.tracking.sort import (
    cxcyah_to_xyxy,
    iou_batch,
    linear_assignment,
    xyxy_to_cxcyah,
)


def cosine_distance(
    track_features: List[np.ndarray], detection_features: np.ndarray
) -> np.ndarray:
    if len(track_features) == 0 or detection_features.shape[0] == 0:
        return np.empty((len(track_features), detection_features.shape[0]), dtype=float)

    if detection_features.size == 0 or (
        detection_features.ndim > 1 and detection_features.shape[1] == 0
    ):
        return np.full(
            (len(track_features), detection_features.shape[0]), 1e5, dtype=float
        )

    cost_matrix = np.zeros(
        (len(track_features), detection_features.shape[0]), dtype=float
    )

    for i, gallery in enumerate(track_features):
        if gallery.size == 0:
            cost_matrix[i, :] = 1.0
            continue
        # Both gallery and detection features are L2-normalized: dot product is cosine similarity
        similarities = np.dot(gallery, detection_features.T)
        max_sim = np.max(similarities, axis=0)
        cost_matrix[i, :] = 1.0 - max_sim

    return cost_matrix


def mahalanobis_gating(
    kf: KalmanFilter,
    tracks: List["DeepTrack"],
    measurements: np.ndarray,
    cost_matrix: np.ndarray,
    gating_threshold: float = 9.4877,
) -> np.ndarray:
    if cost_matrix.size == 0 or len(tracks) == 0 or len(measurements) == 0:
        return cost_matrix

    gated_cost = cost_matrix.copy()

    for row, track in enumerate(tracks):
        proj_mean, proj_cov = kf.project(track.mean, track.covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            proj_cov, lower=True, check_finite=False
        )

        diff = measurements - proj_mean
        z = scipy.linalg.cho_solve((chol_factor, lower), diff.T, check_finite=False)
        squared_distances = np.sum(diff.T * z, axis=0)

        # Invalidate assignments exceeding chi-square confidence threshold
        mask = squared_distances > gating_threshold
        gated_cost[row, mask] = 1e5

    return gated_cost


class DeepTrack:
    count: int = 0

    def __init__(
        self,
        bbox_xyxy: np.ndarray,
        feature: Optional[np.ndarray],
        score: float = 1.0,
        nn_budget: int = 100,
    ) -> None:
        self.kf = KalmanFilter()
        measurement = xyxy_to_cxcyah(bbox_xyxy)
        self.mean, self.covariance = self.kf.initiate(measurement)

        DeepTrack.count += 1
        self.id: int = DeepTrack.count
        self.hits: int = 1
        self.age: int = 0
        self.time_since_update: int = 0
        self.score: float = float(score)
        self.is_confirmed: bool = False
        self.nn_budget = nn_budget

        self.features: deque = deque(maxlen=self.nn_budget)
        if feature is not None and feature.size > 0:
            self.features.append(feature.reshape(-1))

    def predict(self) -> None:
        if self.mean[3] + self.mean[7] <= 0:
            self.mean[7] = 0.0

        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        bbox_xyxy: np.ndarray,
        feature: Optional[np.ndarray],
        score: float = 1.0,
    ) -> None:
        measurement = xyxy_to_cxcyah(bbox_xyxy)
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, measurement
        )
        self.hits += 1
        self.time_since_update = 0
        self.score = float(score)

        if feature is not None and feature.size > 0:
            self.features.append(feature.reshape(-1))

    def get_state(self) -> np.ndarray:
        return cxcyah_to_xyxy(self.mean[:4])

    def get_gallery(self) -> np.ndarray:
        if not self.features:
            return np.empty((0, 512), dtype=np.float32)
        return np.array(self.features, dtype=np.float32)


class DeepSort:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        gating_threshold: float = 9.4877,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.gating_threshold = gating_threshold
        self.kf = KalmanFilter()

        self.tracks: List[DeepTrack] = []
        self.frame_count: int = 0

    @classmethod
    def reset_id_counter(cls) -> None:
        DeepTrack.count = 0

    def update(
        self,
        dets_xyxy: np.ndarray,
        features: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.frame_count += 1

        if dets_xyxy is None or dets_xyxy.size == 0:
            dets_xyxy = np.empty((0, 4), dtype=float)
            num_dets = 0
        else:
            num_dets = dets_xyxy.shape[0]

        if scores is None or scores.size == 0:
            scores = np.ones(num_dets, dtype=float)

        if features is None or features.size == 0:
            features = np.zeros((num_dets, 0), dtype=np.float32)

        # 1. Predict all tracks
        for track in self.tracks:
            track.predict()

        # 2. Separate tracks into confirmed and unconfirmed
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed]
        unconfirmed_tracks = [t for t in self.tracks if not t.is_confirmed]

        # 3. Matching cascade by age for confirmed tracks
        matches_cascade: List[Tuple[DeepTrack, int]] = []
        unmatched_dets = set(range(num_dets))
        unmatched_confirmed_tracks: List[DeepTrack] = []

        if confirmed_tracks and num_dets > 0:
            measurements = np.array([xyxy_to_cxcyah(box) for box in dets_xyxy])

            for age in range(1, self.max_age + 1):
                tracks_at_age = [
                    t for t in confirmed_tracks if t.time_since_update == age
                ]
                if not tracks_at_age or not unmatched_dets:
                    continue

                det_indices = sorted(list(unmatched_dets))
                sub_dets_feat = features[det_indices]
                sub_measurements = measurements[det_indices]

                # Compute appearance cosine distance matrix
                galleries = [t.get_gallery() for t in tracks_at_age]
                cost_mat = cosine_distance(galleries, sub_dets_feat)

                # Gate using Kalman Mahalanobis distance
                cost_mat = mahalanobis_gating(
                    self.kf,
                    tracks_at_age,
                    sub_measurements,
                    cost_mat,
                    self.gating_threshold,
                )

                matches_sub, u_trks_sub, u_dets_sub = linear_assignment(
                    cost_mat, threshold=self.max_cosine_distance
                )

                for trk_idx, det_sub_idx in matches_sub:
                    actual_det_idx = det_indices[det_sub_idx]
                    matches_cascade.append((tracks_at_age[trk_idx], actual_det_idx))
                    unmatched_dets.remove(actual_det_idx)

            matched_confirmed_set = {t for t, _ in matches_cascade}
            unmatched_confirmed_tracks = [
                t for t in confirmed_tracks if t not in matched_confirmed_set
            ]
        else:
            unmatched_confirmed_tracks = confirmed_tracks

        # 4. IoU fallback association
        # Includes unconfirmed tracks and confirmed tracks with time_since_update == 1
        iou_track_candidates = [
            t for t in unmatched_confirmed_tracks if t.time_since_update == 1
        ] + unconfirmed_tracks

        matches_iou: List[Tuple[DeepTrack, int]] = []
        unmatched_iou_tracks: List[DeepTrack] = []

        if iou_track_candidates and unmatched_dets:
            det_indices_iou = sorted(list(unmatched_dets))
            sub_dets_boxes = dets_xyxy[det_indices_iou]
            trk_boxes = np.array([t.get_state() for t in iou_track_candidates])

            iou_cost = 1.0 - iou_batch(trk_boxes, sub_dets_boxes)
            matches_iou_sub, u_trks_iou, u_dets_iou = linear_assignment(
                iou_cost, threshold=1.0 - self.iou_threshold
            )

            for trk_idx, det_sub_idx in matches_iou_sub:
                actual_det_idx = det_indices_iou[det_sub_idx]
                matches_iou.append((iou_track_candidates[trk_idx], actual_det_idx))
                unmatched_dets.remove(actual_det_idx)

            matched_iou_set = {t for t, _ in matches_iou}
            unmatched_iou_tracks = [
                t for t in iou_track_candidates if t not in matched_iou_set
            ]
        else:
            unmatched_iou_tracks = iou_track_candidates

        # 5. Update matched tracks
        all_matches = matches_cascade + matches_iou
        for track, det_idx in all_matches:
            feat = features[det_idx] if features.size > 0 else None
            track.update(dets_xyxy[det_idx], feat, score=scores[det_idx])
            if not track.is_confirmed and track.hits >= self.min_hits:
                track.is_confirmed = True

        # 6. Initialize new tracks for unmatched detections
        for det_idx in unmatched_dets:
            feat = features[det_idx] if features.size > 0 else None
            new_trk = DeepTrack(
                dets_xyxy[det_idx],
                feat,
                score=scores[det_idx],
                nn_budget=self.nn_budget,
            )
            # Immediate confirmation if min_hits is 1
            if self.min_hits <= 1:
                new_trk.is_confirmed = True
            self.tracks.append(new_trk)

        # 7. Remove dead tracks
        surviving: List[DeepTrack] = []
        for track in self.tracks:
            # Unconfirmed tracks with a miss are removed immediately
            if not track.is_confirmed and track.time_since_update > 0:
                continue
            if track.time_since_update > self.max_age:
                continue
            surviving.append(track)
        self.tracks = surviving

        # 8. Output active confirmed tracks updated in this frame
        active_outputs: List[np.ndarray] = []
        for track in self.tracks:
            if (
                track.is_confirmed or self.frame_count <= self.min_hits
            ) and track.time_since_update == 0:
                box = track.get_state()
                active_outputs.append(
                    np.array([box[0], box[1], box[2], box[3], track.id, track.score])
                )

        if active_outputs:
            return np.array(active_outputs, dtype=float)
        return np.empty((0, 6), dtype=float)
