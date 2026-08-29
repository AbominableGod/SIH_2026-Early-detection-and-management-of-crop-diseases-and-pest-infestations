from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np
import io
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tomato_model.h5"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Tomato Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = [
    "Bacterial spot", "Early blight", "Healthy", "Late blight", "Leaf Mold",
    "Septoria leaf spot", "Spider mites", "Target Spot", "Yellow Leaf Curl Virus", "Mosaic virus"
]

model = None

# ALL 10 DISEASES IN 3 LANGUAGES
TRANSLATIONS = {
    "Bacterial spot": {
        "en": {"name": "Bacterial Spot", "treatment": "Use Copper Oxychloride 0.25%. Remove infected leaves.", "prevention": "Avoid overhead watering. Use disease-free seeds."},
        "hi": {"name": "जीवाणु धब्बा", "treatment": "कॉपर ऑक्सीक्लोराइड 0.25% का छिड़काव करें। संक्रमित पत्ते हटा दें।", "prevention": "ऊपर से पानी देने से बचें। रोग मुक्त बीजों का उपयोग करें।"},
        "mr": {"name": "जीवाणू डाग", "treatment": "कॉपर ऑक्सिक्लोराईड 0.25% फवारणी करा. बाधित पाने काढून टाका.", "prevention": "वरून पाणी देणे टाळा. रोगमुक्त बियाणे वापरा."}
    },
    "Early blight": {
        "en": {"name": "Early Blight", "treatment": "Spray Mancozeb 0.2% every 10 days. Remove lower leaves.", "prevention": "Use mulching. Ensure good air circulation."},
        "hi": {"name": "अर्ली ब्लाइट", "treatment": "हर 10 दिन में मैन्कोजेब 0.2% का छिड़काव करें। निचली पत्तियां हटाएं।", "prevention": "मल्चिंग का प्रयोग करें। हवा का आवागमन सुनिश्चित करें।"},
        "mr": {"name": "अर्ली ब्लाइट", "treatment": "दर 10 दिवसांनी मॅन्कोझेब 0.2% फवारणी करा. खालची पाने काढा.", "prevention": "मल्चिंग वापरा. हवेची योग्य देवाणघेवाण सुनिश्चित करा."}
    },
    "Late blight": {
        "en": {"name": "Late Blight", "treatment": "Spray Metalaxyl + Mancozeb 0.25% immediately. Destroy infected plants.", "prevention": "Avoid wet conditions. Preventive spray before monsoon."},
        "hi": {"name": "लेट ब्लाइट", "treatment": "तुरंत मेटालैक्सिल + मैन्कोजेब 0.25% का छिड़काव करें। संक्रमित पौधे नष्ट करें।", "prevention": "गीली स्थिति से बचें। मानसून से पहले रोकथाम के लिए छिड़काव करें।"},
        "mr": {"name": "लेट ब्लाइट", "treatment": "तात्काळ मेटालॅक्सिल + मॅन्कोझेब 0.25% फवारणी करा. बाधित झाडे नष्ट करा.", "prevention": "ओल्या परिस्थिती टाळा. मान्सूनपूर्वी प्रतिबंधात्मक फवारणी करा."}
    },
    "Leaf Mold": {
        "en": {"name": "Leaf Mold", "treatment": "Spray Chlorothalonil 0.2%. Reduce greenhouse humidity.", "prevention": "Improve ventilation. Avoid working when leaves are wet."},
        "hi": {"name": "पत्ती का फफूंद", "treatment": "क्लोरोथॅलोनिल 0.2% का छिड़काव करें। ग्रीनहाउस की नमी कम करें।", "prevention": "हवादारी में सुधार करें। पत्ते गीले होने पर काम न करें।"},
        "mr": {"name": "पान बुरशी", "treatment": "क्लोरोथॅलोनिल 0.2% फवारणी करा. ग्रीनहाऊसमधील आर्द्रता कमी करा.", "prevention": "हवेशीरपणा सुधारा. पाने ओली असताना काम करू नका."}
    },
    "Septoria leaf spot": {
        "en": {"name": "Septoria Leaf Spot", "treatment": "Spray Mancozeb 0.2% every 7-10 days.", "prevention": "Remove plant debris. Use drip irrigation."},
        "hi": {"name": "सेप्टोरिया पत्ती धब्बा", "treatment": "हर 7-10 दिन में मैन्कोजेब 0.2% का छिड़काव करें।", "prevention": "पौधों का कचरा हटाएं। ड्रिप सिंचाई का प्रयोग करें।"},
        "mr": {"name": "सेप्टोरिया पान ठिपके", "treatment": "दर 7-10 दिवसांनी मॅन्कोझेब 0.2% फवारणी करा.", "prevention": "झाडांचा कचरा काढा. ठिबक सिंचन वापरा."}
    },
    "Spider mites": {
        "en": {"name": "Spider Mites", "treatment": "Spray Neem oil 5ml/L or Abamectin 0.5ml/L.", "prevention": "Maintain humidity >60%. Use yellow sticky traps."},
        "hi": {"name": "मकड़ी के कण", "treatment": "नीम तेल 5ml/L या एबामेक्टिन 0.5ml/L का छिड़काव करें।", "prevention": "नमी 60% से ऊपर रखें। पीले चिपचिपे जाल का प्रयोग करें।"},
        "mr": {"name": "कोळी कीड", "treatment": "नीम तेल 5ml/L किंवा अबामेक्टिन 0.5ml/L फवारणी करा.", "prevention": "आर्द्रता 60% पेक्षा जास्त ठेवा. पिवळे चिकट सापळे वापरा."}
    },
    "Target Spot": {
        "en": {"name": "Target Spot", "treatment": "Spray Chlorothalonil 0.2% or Tebuconazole 0.1%.", "prevention": "Avoid dense planting. Control weeds."},
        "hi": {"name": "टारगेट स्पॉट", "treatment": "क्लोरोथॅलोनिल 0.2% या टेबुकोनाजोल 0.1% का छिड़काव करें।", "prevention": "सघन रोपण से बचें। खरपतवार नियंत्रित करें।"},
        "mr": {"name": "टार्गेट स्पॉट", "treatment": "क्लोरोथॅलोनिल 0.2% किंवा टेबुकोनाझोल 0.1% फवारणी करा.", "prevention": "दाट लागवड टाळा. तण नियंत्रण करा."}
    },
    "Yellow Leaf Curl Virus": {
        "en": {"name": "Yellow Leaf Curl Virus", "treatment": "No cure. Uproot and destroy infected plants.", "prevention": "Control whiteflies using Imidacloprid 0.3ml/L + Yellow traps."},
        "hi": {"name": "पीला पत्ता कर्ल वायरस", "treatment": "कोई इलाज नहीं। संक्रमित पौधों को उखाड़कर नष्ट करें।", "prevention": "इमिडाक्लोप्रिड 0.3ml/L + पीले जाल से सफेद मक्खी नियंत्रित करें।"},
        "mr": {"name": "पिवळा पान कुरळे विषाणू", "treatment": "उपचार नाही. बाधित झाडे उपटून नष्ट करा.", "prevention": "इमिडाक्लोप्रिड 0.3ml/L + पिवळे सापळे वापरून पांढरी माशी नियंत्रित करा."}
    },
    "Mosaic virus": {
        "en": {"name": "Mosaic Virus", "treatment": "No cure. Remove infected plants. Control aphids.", "prevention": "Use virus-free seeds. Disinfect tools."},
        "hi": {"name": "मोज़ेक वायरस", "treatment": "कोई इलाज नहीं। संक्रमित पौधे हटाएं। माहू को नियंत्रित करें।", "prevention": "वायरस मुक्त बीज प्रयोग करें। औजारों को कीटाणुरहित करें।"},
        "mr": {"name": "मोझॅक विषाणू", "treatment": "उपचार नाही. बाधित झाडे काढा. मावा नियंत्रित करा.", "prevention": "विषाणू-मुक्त बियाणे वापरा. साधने निर्जंतुक करा."}
    },
    "Healthy": {
        "en": {"name": "Healthy Plant", "treatment": "Plant is healthy. Continue current care.", "prevention": "Regular scouting. Balanced fertilization."},
        "hi": {"name": "स्वस्थ पौधा", "treatment": "पौधा स्वस्थ है। वर्तमान देखभाल जारी रखें।", "prevention": "नियमित जांच करें। संतुलित उर्वरक दें।"},
        "mr": {"name": "निरोगी रोप", "treatment": "झाड निरोगी आहे. सध्याची काळजी सुरू ठेवा.", "prevention": "नियमित तपासणी करा. संतुलित खत द्या."}
    }
}


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="The model is not available. Run train_model.py first.")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "en"):
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Please upload an image smaller than 10 MB.")
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Please upload a valid image file.") from error

    # Resize using PIL
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = get_model().predict(image_array, verbose=0)[0]

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])

    disease = CLASS_NAMES[class_index]
    translation_data = TRANSLATIONS.get(disease, TRANSLATIONS["Healthy"])
    lang_data = translation_data.get(lang, translation_data.get("en"))

    return {
        "predicted_disease": disease,
        "disease_name": lang_data["name"],
        "confidence": round(confidence * 100, 2),
        "treatment": lang_data["treatment"],
        "prevention": lang_data["prevention"]
    }