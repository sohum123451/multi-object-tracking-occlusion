from typing import List, Tuple
import numpy as np

from src.tracker.basetrack import BaseTrack, STrack, TrackState
from src.tracker.kalman_filter import KalmanFilter
from src.tracker.matching import iou_distance, linear_assignment


class ByteTracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        match_thresh_second: float = 0.5,
        min_box_area: float = 10.0,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.match_thresh_second = match_thresh_second
        self.max_time_lost = track_buffer
        self.min_box_area = min_box_area
        self.kalman_filter = KalmanFilter()

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id: int = 0
        BaseTrack.reset_counter()

    def update(self, output_results: np.ndarray) -> List[STrack]:
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        if output_results.size > 0:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]

            remain_inds = scores >= self.track_thresh
            low_inds = (scores < self.track_thresh) & (scores > 0.1)

            dets_first = bboxes[remain_inds]
            scores_first = scores[remain_inds]

            dets_second = bboxes[low_inds]
            scores_second = scores[low_inds]
        else:
            dets_first = np.empty((0, 4), dtype=float)
            scores_first = np.empty((0,), dtype=float)
            dets_second = np.empty((0, 4), dtype=float)
            scores_second = np.empty((0,), dtype=float)

        detections_first: List[STrack] = []
        if len(dets_first) > 0:
            for bbox, score in zip(dets_first, scores_first):
                tlwh = np.array(
                    [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    dtype=float,
                )
                if tlwh[2] * tlwh[3] > self.min_box_area:
                    detections_first.append(STrack(tlwh, score))

        detections_second: List[STrack] = []
        if len(dets_second) > 0:
            for bbox, score in zip(dets_second, scores_second):
                tlwh = np.array(
                    [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    dtype=float,
                )
                if tlwh[2] * tlwh[3] > self.min_box_area:
                    detections_second.append(STrack(tlwh, score))

        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool: List[STrack] = tracked_stracks + self.lost_stracks
        for track in strack_pool:
            track.predict()

        # Step 1: Associate tracked pool with high-score detections
        if len(strack_pool) > 0 and len(detections_first) > 0:
            pool_boxes = np.array([t.tlbr for t in strack_pool])
            det_boxes = np.array([d.tlbr for d in detections_first])
            cost_matrix = iou_distance(pool_boxes, det_boxes)
            matches_first, u_track_first, u_det_first = linear_assignment(
                cost_matrix, threshold=self.match_thresh
            )
        else:
            matches_first = np.empty((0, 2), dtype=int)
            u_track_first = np.arange(len(strack_pool), dtype=int)
            u_det_first = np.arange(len(detections_first), dtype=int)

        for itracked, idet in matches_first:
            track = strack_pool[itracked]
            det = detections_first[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 2: Associate remaining tracks with low-score detections
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track_first
            if strack_pool[i].state == TrackState.Tracked
        ]
        if len(r_tracked_stracks) > 0 and len(detections_second) > 0:
            r_boxes = np.array([t.tlbr for t in r_tracked_stracks])
            det_boxes_second = np.array([d.tlbr for d in detections_second])
            cost_matrix_second = iou_distance(r_boxes, det_boxes_second)
            matches_second, u_track_second, _ = linear_assignment(
                cost_matrix_second, threshold=self.match_thresh_second
            )
        else:
            matches_second = np.empty((0, 2), dtype=int)
            u_track_second = np.arange(len(r_tracked_stracks), dtype=int)

        for itracked, idet in matches_second:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track_second:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 3: Associate unconfirmed tracks with unmatched high-score detections
        remain_dets = [detections_first[i] for i in u_det_first]
        if len(unconfirmed) > 0 and len(remain_dets) > 0:
            unc_boxes = np.array([t.tlbr for t in unconfirmed])
            det_boxes_remain = np.array([d.tlbr for d in remain_dets])
            cost_matrix_unc = iou_distance(unc_boxes, det_boxes_remain)
            matches_unc, u_unc, u_det_remain = linear_assignment(
                cost_matrix_unc, threshold=0.7
            )
        else:
            matches_unc = np.empty((0, 2), dtype=int)
            u_unc = np.arange(len(unconfirmed), dtype=int)
            u_det_remain = np.arange(len(remain_dets), dtype=int)

        for itracked, idet in matches_unc:
            unconfirmed[itracked].update(remain_dets[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unc:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: Initialize new tracks from remaining high-score detections
        for it in u_det_remain:
            track = remain_dets[it]
            # Minimum activation threshold to avoid spurious single-frame false alarms
            if track.score >= self.track_thresh:
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

        # Step 5: Update lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks.extend(activated_stracks)
        self.tracked_stracks.extend(refind_stracks)

        # Maintain unique tracks in tracked pool
        tracked_ids = set()
        dedup_tracked = []
        for t in self.tracked_stracks:
            if t.track_id not in tracked_ids:
                tracked_ids.add(t.track_id)
                dedup_tracked.append(t)
        self.tracked_stracks = dedup_tracked

        self.lost_stracks = [
            t for t in self.lost_stracks if t.state == TrackState.Lost
        ]
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = [
            t for t in self.lost_stracks if t not in self.tracked_stracks
        ]

        output_stracks = [t for t in self.tracked_stracks if t.is_activated]
        return output_stracks
