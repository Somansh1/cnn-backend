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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)[0]
    confidence = float(np.max(predictions))
    label = int(np.argmax(predictions))  # You can map this to class names

    return {"label": label, "confidence": confidence}
