from ultralytics import YOLO
import cv2


class YOLOv12Model:
    def __init__(self, weights_path):
        self.model = YOLO("yolov8n.pt")

    def predict(self, img):
        if img is None:
            return []

        results = self.model(img)

        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                detections.append({
                    "class_id": cls,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        return detections


def draw_boxes(img, detections):
    img = img.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]

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