from typing import List
import numpy as np

from src.tracker.basetrack import BaseTrack, STrack, TrackState
from src.tracker.kalman_filter import KalmanFilter
from src.tracker.matching import iou_distance, linear_assignment


class SortTracker:
    def __init__(
        self,
        det_thresh: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.kalman_filter = KalmanFilter()
        self.tracked_stracks: List[STrack] = []
        self.frame_id: int = 0
        BaseTrack.reset_counter()

    def update(self, output_results: np.ndarray) -> List[STrack]:
        self.frame_id += 1

        if output_results.size > 0:
            scores = output_results[:, 4]
            remain_inds = scores >= self.det_thresh
            dets = output_results[remain_inds]
        else:
            dets = np.empty((0, 5), dtype=float)

        detections: List[STrack] = []
        if len(dets) > 0:
            for det in dets:
                tlbr = det[:4]
                tlwh = np.array(
                    [tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]],
                    dtype=float,
                )
                detections.append(STrack(tlwh, det[4]))

        for track in self.tracked_stracks:
            track.predict()

        if len(self.tracked_stracks) > 0 and len(detections) > 0:
            track_boxes = np.array([t.tlbr for t in self.tracked_stracks])
            det_boxes = np.array([d.tlbr for d in detections])
            cost_matrix = iou_distance(track_boxes, det_boxes)
            matches, u_track, u_det = linear_assignment(
                cost_matrix, threshold=1.0 - self.iou_threshold
            )
        else:
            matches = np.empty((0, 2), dtype=int)
            u_track = np.arange(len(self.tracked_stracks), dtype=int)
            u_det = np.arange(len(detections), dtype=int)

        for itracked, idet in matches:
            track = self.tracked_stracks[itracked]
            det = detections[idet]
            track.update(det, self.frame_id)

        for idet in u_det:
            track = detections[idet]
            track.activate(self.kalman_filter, self.frame_id)
            self.tracked_stracks.append(track)

        updated_tracks: List[STrack] = []
        for i, track in enumerate(self.tracked_stracks):
            if i in u_track:
                track.time_since_update += 1

            if track.time_since_update > self.max_age:
                track.mark_removed()
            else:
                updated_tracks.append(track)

        self.tracked_stracks = updated_tracks

        output_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            # Output tracks that are freshly matched or have met the minimum hit persistence
            if track.time_since_update == 0 and (
                track.tracklet_len >= self.min_hits or self.frame_id <= self.min_hits
            ):
                output_stracks.append(track)

        return output_stracks
