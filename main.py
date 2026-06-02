from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and class indices once when server starts
model = tf.keras.models.load_model("plant_disease_model.keras")
class_indices = json.load(open("class_indices.json"))

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    img = Image.open(io.BytesIO(await file.read())).resize((224, 224))
    img_array = np.expand_dims(np.array(img).astype('float32') / 255., axis=0)
    
    # Run prediction
    prediction = model.predict(img_array)
    predicted_class = class_indices[str(np.argmax(prediction))]
    confidence = float(np.max(prediction)) * 100
    
    return {
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }