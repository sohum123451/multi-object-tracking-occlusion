from pathlib import Path
from typing import Union
import numpy as np
import torch
from ultralytics import YOLO


class YOLODetector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(self.weights)

    def detect_frame(
        self,
        image_input: Union[Path, str, np.ndarray],
    ) -> np.ndarray:
        # Pretrained model inference filtered strictly to class 0 (person)
        results = self.model.predict(
            source=str(image_input) if isinstance(image_input, Path) else image_input,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[0],
            device=self.device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return np.empty((0, 6), dtype=float)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=float)

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy()

        is_person = clses == 0
        xyxy = xyxy[is_person]
        scores = scores[is_person]
        clses = clses[is_person]

        if len(xyxy) == 0:
            return np.empty((0, 6), dtype=float)

        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]

        return np.column_stack([x1, y1, w, h, scores, clses])
