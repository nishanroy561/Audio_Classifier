# train.py
import os
import numpy as np
import tensorflow as tf
from model import build_cnn_model

def load_data(features_dir, labels, num_classes):
    X = []
    y = []
    
    for root, dirs, files in os.walk(features_dir):
        for filename in files:
            if filename.endswith('.npy'):
                feature_path = os.path.join(root, filename)
                try:
                    feature = np.load(feature_path)
                    X.append(feature)
                    
                    # Extract the label from the file path
                    label = os.path.basename(root)  # e.g., 'rock', 'jazz', 'classical'
                    if label in labels:
                        y.append(labels[label])
                    else:
                        print(f"Label {label} not found in labels dictionary.")
                
                except Exception as e:
                    print(f"Error loading {feature_path}: {e}")
    
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes)
    
    # Debugging: Print the shape of X
    print(f"Shape of X before reshaping: {X.shape}")
    
    # Check if reshaping is needed
    if len(X.shape) == 3:  # (num_samples, feature_dim, time_steps)
        print("X already in the correct shape, no reshaping needed.")
    else:
        raise ValueError("Unexpected shape of X. Check the feature extraction process.")
    
    return X, y

def train_model():
    features_dir = 'data/features/genre'
    labels = {
        "classical": 0,
        "jazz": 1,
        "rock": 2
    }
    num_classes = 3
    input_shape = (13, 100, 1)  # Example input shape for MFCCs
    
    X, y = load_data(features_dir, labels, num_classes)
    
    model = build_cnn_model(input_shape, num_classes)
    
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save the trained model weights
    model.save('models/model_weights.h5')
    
if __name__ == '__main__':
    train_model()
