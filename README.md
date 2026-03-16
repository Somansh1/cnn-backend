# Apple Leaf Disease Detection — Backend

Flask inference API for a custom CNN that classifies apple leaf diseases across 4 classes from the [PlantVillage dataset](https://www.tensorflow.org/datasets/catalog/plant_village). Accepts an image, runs it through a Keras `.h5` model, and returns the predicted class with per-class confidence scores.

**Frontend repo:** [cnn-frontend](https://github.com/Somansh1/cnn-frontend)  
**Live demo:** [app-dlite-net.vercel.app](https://app-dlite-net.vercel.app)

---

## Model

| Property | Value |
|---|---|
| Architecture | Custom CNN (built from scratch) |
| Dataset | PlantVillage |
| Classes | 4 (diseases + healthy) |
| Accuracy | 99.16% on test set |
| Model size | 8 MB (`.h5`) |
| Framework | TensorFlow / Keras |

## API

### `POST /predict`

Accepts a leaf image and returns the top predicted class with confidence scores.

**Request**
```
Content-Type: multipart/form-data
Body: image (file)
```

**Response**
```json
{
  "predicted_class": "Apple___Apple_scab",
  "confidence": 0.9812,
  "all_scores": {
    "Apple___Apple_scab": 0.9812,
    "Apple___Black_rot": 0.0103,
    ...
  }
}
```

### `GET /health`

Returns `200 OK` if the service is running and the model is loaded.

## Getting started

**Prerequisites:** Python 3.8+, pip

```bash
git clone https://github.com/Somansh1/cnn-backend.git
cd cnn-backend
pip install -r requirements.txt
```

Place your trained model file in the project root (or update the path in `app.py`):

```
cnn-backend/
├── app.py
├── model.h5          # ← your trained Keras model
├── requirements.txt
├── Procfile
└── scripts/
```

```bash
python app.py         # starts on http://localhost:5000
```

## Deployment

Configured for Heroku via `Procfile`:

```bash
heroku create your-app-name
git push heroku main
```

The `scripts/` directory contains shell helpers for environment setup.

## Related

- **Frontend:** React app with image upload and confidence score display — [cnn-frontend](https://github.com/Somansh1/cnn-frontend)
- **Training notebook:** Available on request

## License

MIT
