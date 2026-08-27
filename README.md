# Tomato Leaf Disease Detection

FastAPI and TensorFlow MobileNetV2 application for classifying ten tomato leaf conditions.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model. This downloads the Kaggle dataset automatically and creates `tomato_model.h5`:

```bash
python train_model.py
```

Start the API:

```bash
uvicorn app:app --reload
```

Open http://localhost:8000 and upload a leaf image. API documentation is available at http://localhost:8000/docs.

The current workspace already contains `tomato_model.h5`, so training is only needed when retraining the model.