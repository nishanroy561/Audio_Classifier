from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import os
import numpy as np
import tensorflow as tf
import librosa
from typing import List

app = FastAPI()

# Load the trained model globally
model = None

def load_model(model_path):
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path)
    return model

def extract_mfcc(audio_path, n_mfcc=13, max_length=100):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] > max_length:
        mfcc = mfcc[:, :max_length]
    elif mfcc.shape[1] < max_length:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    
    return mfcc

def predict_audio(model, audio_path, n_mfcc=13, max_length=100):
    features = extract_mfcc(audio_path, n_mfcc, max_length)
    features = np.expand_dims(features, axis=0)
    
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return predicted_class

# Labels for the model predictions
labels = {
    0: "classical",
    1: "jazz",
    2: "rock"
}

@app.on_event("startup")
async def startup_event():
    # Load model on startup
    load_model('models/model_weights.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    upload_dir = "audio_app/"
    os.makedirs(upload_dir, exist_ok=True)
    audio_path = os.path.join(upload_dir, file.filename)
    
    with open(audio_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Perform prediction
    predicted_class = predict_audio(model, audio_path)
    predicted_label = labels.get(predicted_class, "Unknown")
    
    # Clean up the saved file
    os.remove(audio_path)
    
    return {"filename": file.filename, "predicted_genre": predicted_label}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
