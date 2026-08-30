markdown# 🌱 SIH 2026: Early Detection and Management of Crop Diseases

AI-powered Crop Disease Detection system for Tomato plants using Deep Learning + FastAPI + Multi-language Support.

## 🚀 Features
- **AI Model**: TensorFlow CNN for Tomato Leaf Disease Classification
- **10 Classes**: Detects 9 diseases + Healthy 
  `Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria, Spider mites, Target Spot, YLCV, Mosaic, Healthy`
- **Multi-language**: English, Hindi, Marathi support
- **Treatment & Prevention**: Instant advisory from database
- **Dark/Light Mode**: Farmer-friendly UI
- **FastAPI Backend**: 1 API call = Prediction in <2 seconds

## 🛠️ Tech Stack
`Python` `TensorFlow 2.21` `FastAPI` `Uvicorn` `Pillow` `HTML/CSS/JS`

## 📂 Project StructureSIH_2026-Early-detection-and-management-of-crop-diseases/
├── app.py                 # FastAPI backend
├── index.html             # Frontend UI
├── model/
│   └── crop_disease_model.h5  # Pre-trained TF model
├── advisories.json        # Treatment + Prevention data
├── requirements.txt       # Dependencies
└── README.mdjavascript
## ⚡ How to Run
1.  **Install dependencies**
    ```bash
    pip install -r requirements.txtStart Serverbash    uvicorn app:app --reloadOpen in Browserjavascript    http://localhost:8000Upload a tomato leaf image and get instant disease detection + treatment in your language.
API documentation: http://localhost:8000/docs
📡 API EndpointsMethodEndpointDescriptionGET/healthCheck server + model statusPOST/predict?lang=hiUpload image + get disease + advisoryGET/Frontend UI👨‍💻 Team
SIH 2026 - Problem Statement: Early detection and management of crop diseases and pest infestations
