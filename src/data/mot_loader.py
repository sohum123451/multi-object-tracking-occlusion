import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class Detection:
    frame: int
    track_id: int
    bbox_tlwh: np.ndarray
    score: float

    @property
    def bbox_tlbr(self) -> np.ndarray:
        x, y, w, h = self.bbox_tlwh
        return np.array([x, y, x + w, y + h], dtype=float)


@dataclass
class GroundTruth:
    frame: int
    track_id: int
    bbox_tlwh: np.ndarray
    conf: float
    class_id: int
    visibility: float

    @property
    def bbox_tlbr(self) -> np.ndarray:
        x, y, w, h = self.bbox_tlwh
        return np.array([x, y, x + w, y + h], dtype=float)


@dataclass
class MOTSequence:
    name: str
    seq_dir: Path
    seq_length: int
    frame_rate: int
    im_width: int
    im_height: int
    detections: List[Detection]
    ground_truths: List[GroundTruth]

    def get_detections_by_frame(self) -> Dict[int, List[Detection]]:
        by_frame: Dict[int, List[Detection]] = {}
        for det in self.detections:
            by_frame.setdefault(det.frame, []).append(det)
        return by_frame

    def get_ground_truths_by_frame(
        self,
        pedestrian_only: bool = True,
        min_visibility: float = 0.0,
    ) -> Dict[int, List[GroundTruth]]:
        by_frame: Dict[int, List[GroundTruth]] = {}
        for gt in self.ground_truths:
            if pedestrian_only and gt.class_id != 1:
                continue
            if gt.visibility < min_visibility:
                continue
            by_frame.setdefault(gt.frame, []).append(gt)
        return by_frame


def tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
    ret = np.asarray(tlwh, dtype=float).copy()
    ret[2] = ret[0] + ret[2]
    ret[3] = ret[1] + ret[3]
    return ret


def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
    ret = np.asarray(tlbr, dtype=float).copy()
    ret[2] = ret[2] - ret[0]
    ret[3] = ret[3] - ret[1]
    return ret


def parse_seqinfo(seq_dir: Path) -> Tuple[int, int, int, int]:
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.exists():
        return 0, 30, 1920, 1080

    config = configparser.ConfigParser()
    config.read(ini_path)
    seq = config["Sequence"] if "Sequence" in config else {}
    seq_length = int(seq.get("seqLength", 0))
    frame_rate = int(seq.get("frameRate", 30))
    im_width = int(seq.get("imWidth", 1920))
    im_height = int(seq.get("imHeight", 1080))
    return seq_length, frame_rate, im_width, im_height


def parse_detections(det_file: Path) -> List[Detection]:
    detections: List[Detection] = []
    if not det_file.exists():
        return detections

    with det_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                parts = line.split()
            if len(parts) < 7:
                continue

            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6])

            detections.append(
                Detection(
                    frame=frame,
                    track_id=track_id,
                    bbox_tlwh=np.array([x, y, w, h], dtype=float),
                    score=score,
                )
            )
    return detections


def parse_ground_truths(gt_file: Path) -> List[GroundTruth]:
    ground_truths: List[GroundTruth] = []
    if not gt_file.exists():
        return ground_truths

    with gt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                parts = line.split()
            if len(parts) < 6:
                continue

            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            class_id = int(float(parts[7])) if len(parts) > 7 else 1
            visibility = float(parts[8]) if len(parts) > 8 else 1.0

            ground_truths.append(
                GroundTruth(
                    frame=frame,
                    track_id=track_id,
                    bbox_tlwh=np.array([x, y, w, h], dtype=float),
                    conf=conf,
                    class_id=class_id,
                    visibility=visibility,
                )
            )
    return ground_truths


def load_sequence(seq_dir: Path) -> MOTSequence:
    name = seq_dir.name
    seq_length, frame_rate, im_width, im_height = parse_seqinfo(seq_dir)
    detections = parse_detections(seq_dir / "det" / "det.txt")
    ground_truths = parse_ground_truths(seq_dir / "gt" / "gt.txt")

    if seq_length == 0:
        max_frame = 0
        if detections:
            max_frame = max(max_frame, max(d.frame for d in detections))
        if ground_truths:
            max_frame = max(max_frame, max(g.frame for g in ground_truths))
        seq_length = max_frame

    return MOTSequence(
        name=name,
        seq_dir=seq_dir,
        seq_length=seq_length,
        frame_rate=frame_rate,
        im_width=im_width,
        im_height=im_height,
        detections=detections,
        ground_truths=ground_truths,
    )
