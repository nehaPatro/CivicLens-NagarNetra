import sys
import os
import cv2
import torch

# 🔥 Use your local yolov12 repo
sys.path.append(os.path.abspath("yolov12"))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device


class YOLOv12Model:
    def __init__(self, weights_path):
        self.device = select_device("cpu")
        self.model = DetectMultiBackend(weights_path, device=self.device)

    def predict(self, img):
        img0 = img.copy()

        img = cv2.resize(img, (640, 640))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        pred = self.model(img)
        pred = non_max_suppression(pred, 0.25, 0.45)

        detections = []

        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(x.item()) for x in xyxy]

                    detections.append({
                        "class_id": int(cls.item()),
                        "confidence": float(conf.item()),
                        "bbox": [x1, y1, x2, y2]
                    })

        return detections


def draw_boxes(img, detections):
    img = img.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"Pothole {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img


def format_detections(detections):
    return {
        "count": len(detections),
        "message": f"{len(detections)} pothole(s) detected" if detections else "No potholes detected",
        "detections": detections
    }