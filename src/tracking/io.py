from pathlib import Path
from typing import Dict, List
import numpy as np


def load_mot_detections(det_path: Path) -> Dict[int, np.ndarray]:
    detections_by_frame: Dict[int, List[List[float]]] = {}
    if not det_path.exists():
        return {}

    with det_path.open("r", encoding="utf-8") as f:
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
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0

            # Convert tlwh to x1, y1, x2, y2, score
            detections_by_frame.setdefault(frame, []).append(
                [x, y, x + w, y + h, conf]
            )

    result: Dict[int, np.ndarray] = {}
    for frame, dets in detections_by_frame.items():
        result[frame] = np.array(dets, dtype=float)
    return result


def format_mot_row(
    frame: int,
    track_id: int,
    bbox_xyxy: np.ndarray,
    conf: float = 1.0,
) -> str:
    x1, y1, x2, y2 = bbox_xyxy[:4]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return f"{frame},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
