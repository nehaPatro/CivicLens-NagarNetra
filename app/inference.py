import sys
import os
import cv2

# Ensure yolov12 repo is accessible
sys.path.append(os.path.abspath("yolov12"))

from ultralytics import YOLO


class YOLOv12Model:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def predict(self, img):
        results = self.model(img)

        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    "class_id": cls,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        return detections


# 🔥 NEW: Draw bounding boxes
def draw_boxes(img, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]

        # Draw RED rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"Pothole {conf:.2f}"

        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    return img


# Optional: readable message
def format_detections(detections):
    if len(detections) == 0:
        return {
            "count": 0,
            "message": "No potholes detected",
            "detections": []
        }

    return {
        "count": len(detections),
        "message": f"{len(detections)} pothole(s) detected",
        "detections": detections
    }