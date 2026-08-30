import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import io, json
from pathlib import Path

with open('advisories.json', 'r', encoding='utf-8') as f:
    ADVISORIES = json.load(f)

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMG_SIZE = (256, 256)

CLASS_NAMES = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato_Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus',
    'Tomato_healthy', 'Tomato_Gray_leaf_spot'
]

CLASS_TO_ADVISORY_KEY = {
    'Tomato_Bacterial_spot': 'Bacterial spot', 'Tomato_Early_blight': 'Early blight',
    'Tomato_Late_blight': 'Late blight', 'Tomato_Leaf_Mold': 'Leaf Mold',
    'Tomato_Septoria_leaf_spot': 'Septoria leaf spot',
    'Tomato_Spider_mites_Two-spotted_spider_mite': 'Spider mites',
    'Tomato_Target_Spot': 'Target Spot',
    'Tomato_Yellow_Leaf_Curl_Virus': 'Yellow Leaf Curl Virus',
    'Tomato_mosaic_virus': 'Mosaic virus', 'Tomato_healthy': 'Healthy',
    'Tomato_Gray_leaf_spot': 'Gray leaf spot'
}

DISEASE_NAMES = {
    'Tomato_Bacterial_spot': {'en': 'Bacterial Spot', 'hi': 'जीवाणु धब्बा', 'mr': 'जीवाणू डाग'},
    'Tomato_Early_blight': {'en': 'Early Blight', 'hi': 'अर्ली ब्लाइट', 'mr': 'अर्ली ब्लाइट'},
    'Tomato_Late_blight': {'en': 'Late Blight', 'hi': 'लेट ब्लाइट', 'mr': 'लेट ब्लाइट'},
    'Tomato_Leaf_Mold': {'en': 'Leaf Mold', 'hi': 'पत्ती का फफूंद', 'mr': 'पान बुरशी'},
    'Tomato_Septoria_leaf_spot': {'en': 'Septoria Leaf Spot', 'hi': 'सेप्टोरिया पत्ती धब्बा', 'mr': 'सेप्टोरिया पान ठिपके'},
    'Tomato_Spider_mites_Two-spotted_spider_mite': {'en': 'Spider Mites', 'hi': 'मकड़ी के कण', 'mr': 'कोळी कीड'},
    'Tomato_Target_Spot': {'en': 'Target Spot', 'hi': 'टारगेट स्पॉट', 'mr': 'टार्गेट स्पॉट'},
    'Tomato_Yellow_Leaf_Curl_Virus': {'en': 'Yellow Leaf Curl Virus', 'hi': 'पीला पत्ता कर्ल वायरस', 'mr': 'पिवळा पान कुरळे विषाणू'},
    'Tomato_mosaic_virus': {'en': 'Mosaic Virus', 'hi': 'मोज़ेक वायरस', 'mr': 'मोझॅक विषाणू'},
    'Tomato_healthy': {'en': 'Healthy Plant', 'hi': 'स्वस्थ पौधा', 'mr': 'निरोगी रोप'},
    'Tomato_Gray_leaf_spot': {'en': 'Gray Leaf Spot', 'hi': 'ग्रे लीफ स्पॉट', 'mr': 'ग्रे लीफ स्पॉट'},
    'Unable_to_identify': {'en': 'Unable to confidently identify disease. Please upload a clear image of a single tomato leaf', 'hi': 'रोग की पहचान नहीं हो पाई। कृपया टमाटर के पत्ते की स्पष्ट छवि अपलोड करें', 'mr': 'रोग ओळखता आला नाही. कृपया टोमॅटोच्या पानाचा स्पष्ट फोटो अपलोड करा' },
}

app = FastAPI(title="Crop Disease Detection API - SIH 2026", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = tf.keras.models.load_model('model/crop_disease_model.keras')
CONFIDENCE_THRESHOLD = 0.70

@app.get("/health")
def health():
    return {"status": "ok", "model_input": str(model.input_shape), "languages": ["en", "hi", "mr"]}

@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "en"):
    if lang not in ["en", "hi", "mr"]: lang = "en"
    if not file.content_type.startswith("image"): raise HTTPException(status_code=400, detail="Please upload a valid image file")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES: raise HTTPException(status_code=413, detail="File too large. Max 10MB")

    try: img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception: raise HTTPException(status_code=400, detail="Invalid image file. Use JPG/PNG")

    if img.size[0] < 100 or img.size[1] < 100: raise HTTPException(status_code=400, detail="Image too small")

    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    confidence = float(np.max(predictions))
    predicted_index = int(np.argmax(predictions))

    print("Predicted class index:", predicted_index)
    print("Disease key:", CLASS_NAMES[predicted_index])

    disease_key = CLASS_NAMES[predicted_index]

    if confidence < CONFIDENCE_THRESHOLD:
        return {"status": "low_confidence", "message": DISEASE_NAMES['Unable_to_identify'].get(lang), "confidence": round(confidence * 100, 2)}

    advisory_key = CLASS_TO_ADVISORY_KEY.get(disease_key, 'Healthy')
    advisory = ADVISORIES.get(advisory_key, ADVISORIES['Healthy'])
    disease_name = DISEASE_NAMES.get(disease_key, DISEASE_NAMES['Tomato_healthy']).get(lang, "Unknown")

    return {"status": "success", "disease_key": disease_key, "disease": disease_name, "confidence": round(confidence * 100, 2), "treatment": advisory.get("treatment", "No data"), "prevention": advisory.get("prevention", "No data"), "symptoms": advisory.get("symptoms", "No data"), "language": lang}

app.mount("/", StaticFiles(directory=".", html=True), name="static")