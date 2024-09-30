# predict.py
import os
import numpy as np
import tensorflow as tf
import librosa

def extract_mfcc(audio_path, n_mfcc=13, max_length=100):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Pad or truncate MFCC to a fixed length
    if mfcc.shape[1] > max_length:
        mfcc = mfcc[:, :max_length]
    elif mfcc.shape[1] < max_length:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    
    return mfcc

def load_model(model_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    return model

def predict_audio(model, audio_path, n_mfcc=13, max_length=100):
    # Extract features from the audio file
    features = extract_mfcc(audio_path, n_mfcc, max_length)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return predicted_class

def predict_directory(model, input_dir, n_mfcc=13, max_length=100):
    labels = {
        0: "classical",
        1: "jazz",
        2: "rock"
    }
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.mp3') or filename.endswith('.wav'):
            audio_path = os.path.join(input_dir, filename)
            predicted_class = predict_audio(model, audio_path, n_mfcc, max_length)
            predicted_label = labels.get(predicted_class, "Unknown")
            print(f"Predicted label for {filename}: {predicted_label}")

def main():
    model_path = 'models/model_weights.h5'  # Path to the saved model
    input_dir = 'Sample'  # Directory containing sample audio files
    
    model = load_model(model_path)
    predict_directory(model, input_dir)

if __name__ == '__main__':
    main()
