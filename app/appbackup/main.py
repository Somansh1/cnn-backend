# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf  # or import torch

app = FastAPI()

# Allow your React front end (running e.g. on http://localhost:3000 or your Heroku URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model once at startup
MODEL_PATH = "model.h5"      # adjust if you have .pt
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    data = await file.read()
    img = Image.open(io.BytesIO(data))
    x = preprocess(img)
    preds = model.predict(x)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    # Map index → label (define your own list)
    labels = ["cat", "dog", "bird"]
    return {"label": labels[class_idx], "confidence": confidence}
