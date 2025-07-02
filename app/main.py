# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Allow requests from all origins (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained CNN model
model = load_model("model.h5")

@app.get("/")
def root():
    return {"message": "CNN Image Classifier API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print(f"✅ File received: {file.filename}, size={len(contents)} bytes")
        return {"message": "File received successfully", "filename": file.filename}
    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return {"error": "Failed to process image"}
