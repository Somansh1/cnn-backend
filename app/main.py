# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO

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
        print(f"📥 File size: {len(contents)}")

        image = Image.open(BytesIO(contents)).convert("RGB")
        print(f"🖼️ Image format: {image.format}, size: {image.size}")

        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = image.reshape(224, 224, 3)
        print(f"✅ Image array shape: {image.shape}")

        prediction = model.predict(image[np.newaxis, ...])
        print(f"🔮 Prediction: {prediction}")

        result = {"class": int(np.argmax(prediction))}
        return result
    except Exception as e:
        print("❌ Error in /predict:", e)
        return {"error": str(e)}
