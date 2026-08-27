import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tomato_model.h5"

CLASS_NAMES = [
    "Bacterial spot", "Early blight", "Healthy", "Late blight", "Leaf Mold",
    "Septoria leaf spot", "Spider mites", "Target Spot",
    "Yellow Leaf Curl Virus", "Mosaic virus",
]

app = FastAPI(title="Tomato Leaf Disease Detection")

# ALLOW FRONTEND TO CALL BACKEND
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}. Run train_model.py first.")

MODEL = tf.keras.models.load_model(MODEL_PATH)

# 3. LOAD TREATMENTS
with (BASE_DIR / "advisories.json").open(encoding="utf-8") as f:
    advisories = json.load(f)

# 4. SERVE FRONTEND
@app.get("/")
async def read_root():
    return FileResponse(BASE_DIR / "index.html")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_array = preprocess_image(await file.read())
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Please upload a valid image file.") from error

    predictions = MODEL.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = f"{np.max(predictions[0])*100:.2f}%"

    advisory = advisories.get(predicted_class, advisories["Healthy"])

    return JSONResponse({
        "crop": "Tomato",
        "predicted_disease": predicted_class,
        "confidence": confidence,
        "treatment": advisory["treatment"],
        "organic_solution": advisory["organic"],
        "weather_alert": {"risk": "Weather advisory unavailable."}
    })