from enum import Enum
from typing import Tuple
import numpy as np

from src.tracker.kalman_filter import KalmanFilter


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count: int = 0

    def __init__(self) -> None:
        self.track_id: int = 0
        self.is_activated: bool = False
        self.state: TrackState = TrackState.New
        self.history: list[np.ndarray] = []
        self.features: list[np.ndarray] = []
        self.curr_feature: np.ndarray | None = None
        self.score: float = 0.0
        self.start_frame: int = 0
        self.frame_id: int = 0
        self.time_since_update: int = 0

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_counter(cls) -> None:
        cls._count = 0

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float) -> None:
        super().__init__()
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.score = float(score)
        self.kalman_filter: KalmanFilter | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.tracklet_len: int = 0

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[0] += ret[2] / 2.0
        ret[1] += ret[3] / 2.0
        ret[2] /= np.maximum(ret[3], 1e-6)
        return ret

    def to_xyah(self) -> np.ndarray:
        return self.tlwh_to_xyah(self.tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[0] -= ret[2] / 2.0
        ret[1] -= ret[3] / 2.0
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh
        ret[2] += ret[0]
        ret[3] += ret[1]
        return ret

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0

    def re_activate(
        self,
        new_track: "STrack",
        frame_id: int,
        new_id: bool = False,
    ) -> None:
        if self.kalman_filter is None or self.mean is None or self.covariance is None:
            raise RuntimeError("Cannot re-activate an uninitialized track")
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.time_since_update = 0
        if new_id:
            self.track_id = self.next_id()

    def predict(self) -> None:
        if self.kalman_filter is None or self.mean is None or self.covariance is None:
            return
        if self.state != TrackState.Tracked:
            # Dampen velocity components when object is not observed
            self.mean[7] = 0.0
        self.mean, self.covariance = self.kalman_filter.predict(
            self.mean, self.covariance
        )

    def update(self, new_track: "STrack", frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        if self.kalman_filter is None or self.mean is None or self.covariance is None:
            raise RuntimeError("Cannot update an uninitialized track")
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.time_since_update = 0
