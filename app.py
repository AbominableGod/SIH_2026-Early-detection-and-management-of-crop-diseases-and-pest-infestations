import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import io, json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "crop_disease_model.h5"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMG_SIZE = (256, 256)

CLASS_NAMES = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

CLASS_TO_ADVISORY_KEY = {
    'Tomato___Bacterial_spot': 'Bacterial spot', 'Tomato___Early_blight': 'Early blight',
    'Tomato___Late_blight': 'Late blight', 'Tomato___Leaf_Mold': 'Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Septoria leaf spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider mites',
    'Tomato___Target_Spot': 'Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Mosaic virus', 'Tomato___healthy': 'Healthy'
}

DISEASE_NAMES = {
    'Tomato___Bacterial_spot': {'en': 'Bacterial Spot', 'hi': 'जीवाणु धब्बा', 'mr': 'जीवाणू डाग'},
    'Tomato___Early_blight': {'en': 'Early Blight', 'hi': 'अर्ली ब्लाइट', 'mr': 'अर्ली ब्लाइट'},
    'Tomato___Late_blight': {'en': 'Late Blight', 'hi': 'लेट ब्लाइट', 'mr': 'लेट ब्लाइट'},
    'Tomato___Leaf_Mold': {'en': 'Leaf Mold', 'hi': 'पत्ती का फफूंद', 'mr': 'पान बुरशी'},
    'Tomato___Septoria_leaf_spot': {'en': 'Septoria Leaf Spot', 'hi': 'सेप्टोरिया पत्ती धब्बा', 'mr': 'सेप्टोरिया पान ठिपके'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'en': 'Spider Mites', 'hi': 'मकड़ी के कण', 'mr': 'कोळी कीड'},
    'Tomato___Target_Spot': {'en': 'Target Spot', 'hi': 'टारगेट स्पॉट', 'mr': 'टार्गेट स्पॉट'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'en': 'Yellow Leaf Curl Virus', 'hi': 'पीला पत्ता कर्ल वायरस', 'mr': 'पिवळा पान कुरळे विषाणू'},
    'Tomato___Tomato_mosaic_virus': {'en': 'Mosaic Virus', 'hi': 'मोज़ेक वायरस', 'mr': 'मोझॅक विषाणू'},
    'Tomato___healthy': {'en': 'Healthy Plant', 'hi': 'स्वस्थ पौधा', 'mr': 'निरोगी रोप'}
}

# ============================================================================
# APP INITIALIZATION - ONLY 1 TIME
# ============================================================================
app = FastAPI(title="Crop Disease Detection API - SIH 2026", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = tf.keras.models.load_model(MODEL_PATH)
ADVISORIES = json.load(open(BASE_DIR / 'advisories.json', 'r', encoding='utf-8'))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(IMG_SIZE, Image.Resampling.LANCZOS)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Use JPG/PNG")
    img_array = tf.keras.utils.img_to_array(img)
    return tf.expand_dims(img_array / 255.0, 0)

def get_lang_data(model_key: str, lang: str) -> dict:
    advisory_key = CLASS_TO_ADVISORY_KEY[model_key]
    advisory = ADVISORIES.get(advisory_key, ADVISORIES.get('Healthy', {}))
    return advisory

# ============================================================================
# API ROUTES
# ============================================================================
@app.get("/health")
def health():
    return {"status": "ok", "model_input": str(model.input_shape), "languages": ["en", "hi", "mr"]}

@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "en"):
    if lang not in ['en', 'hi', 'mr']: lang = 'en'
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)
    pred = model.predict(img_array, verbose=0)
    i = int(np.argmax(pred[0]))
    confidence = float(np.max(pred[0]))
    disease_key = CLASS_NAMES[i]
    lang_data = get_lang_data(disease_key, lang)
    disease_name = DISEASE_NAMES.get(disease_key, DISEASE_NAMES['Tomato___healthy']).get(lang, 'Unknown')
    return {
        "disease_key": disease_key, "disease_name": disease_name, "confidence": round(confidence * 100, 2),
        "treatment": lang_data.get("treatment", "No data"), "prevention": lang_data.get("prevention", "No data"),
        "language": lang
    }

# MOUNT STATIC FILES LAST - THIS SERVES index.html
app.mount("/", StaticFiles(directory=".", html=True), name="static")