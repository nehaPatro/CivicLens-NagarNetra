from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
import json

from app.inference import YOLOv12Model, draw_boxes, format_detections

app = FastAPI()

model = YOLOv12Model("best.pt")


@app.get("/")
def home():
    return {"message": "Pothole Detection API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run model
    detections = model.predict(img)

    # Draw bounding boxes
    output_img = draw_boxes(img, detections)

    # Convert image to bytes
    _, buffer = cv2.imencode(".jpg", output_img)
    io_buf = io.BytesIO(buffer)

    # Format readable output
    formatted = format_detections(detections)

    # Send info in headers
    headers = {
        "Pothole-Detection-Count": str(formatted["count"]),
        "Pothole-Detection-Message": formatted["message"],
        "Pothole-Detections": json.dumps(formatted["detections"])
    }

    return StreamingResponse(io_buf, media_type="image/jpeg", headers=headers)