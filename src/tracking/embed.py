from pathlib import Path
from typing import List, Union
import cv2
import numpy as np
import torch
import torchreid
from torchvision import transforms


class OSNetExtractor:
    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=1000,
            pretrained=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def extract(self, crops: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        if not crops:
            return np.empty((0, 512), dtype=np.float32)

        embeddings_list: List[np.ndarray] = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            tensors = [self.transform(crop) for crop in batch_crops]
            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                features = self.model(batch_tensor)

            features_np = features.cpu().numpy()
            embeddings_list.append(features_np)

        raw_features = np.vstack(embeddings_list)
        # Apply L2-normalization
        norms = np.linalg.norm(raw_features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalized_features = raw_features / norms
        return normalized_features.astype(np.float32)

    def extract_from_frame(
        self,
        frame_bgr: np.ndarray,
        bboxes_xyxy: np.ndarray,
    ) -> np.ndarray:
        if bboxes_xyxy.size == 0:
            return np.empty((0, 512), dtype=np.float32)

        h_img, w_img = frame_bgr.shape[:2]
        crops: List[np.ndarray] = []

        for box in bboxes_xyxy:
            x1 = max(0, int(round(box[0])))
            y1 = max(0, int(round(box[1])))
            x2 = min(w_img, int(round(box[2])))
            y2 = min(h_img, int(round(box[3])))

            # Fallback for degenerate bounding boxes
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            else:
                crop = frame_bgr[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crops.append(crop)

        return self.extract(crops)
