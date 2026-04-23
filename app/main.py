from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
import os

from app.inference import YOLOv12Model, draw_boxes, format_detections

app = FastAPI()

# 🔥 FIXED model path (works on Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "best.pt")

model = YOLOv12Model(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Pothole Detection API Running"}


# 🔴 IMAGE OUTPUT (what you want)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detections = model.predict(img)

        # Draw boxes
        output_img = draw_boxes(img, detections)

        # Convert to image response
        _, buffer = cv2.imencode(".jpg", output_img)
        io_buf = io.BytesIO(buffer)

        return StreamingResponse(io_buf, media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}


# 🟢 TEXT OUTPUT (for understanding)
@app.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detections = model.predict(img)

        return format_detections(detections)

    except Exception as e:
        return {"error": str(e)}