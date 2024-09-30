# extract_features.py
import os
import numpy as np
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

def save_features(audio_path, output_path, n_mfcc=13, max_length=100):
    # Extract features
    mfcc = extract_mfcc(audio_path, n_mfcc, max_length)
    # Save features as .npy file
    np.save(output_path, mfcc)
    print(f"Saved features to {output_path}")

def process_directory(input_dir, output_dir, n_mfcc=13, max_length=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory {output_dir}")
    
    files_processed = 0
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.mp3') or filename.endswith('.wav'):
                audio_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                output_path = os.path.join(output_subdir, filename.replace('.mp3', '.npy').replace('.wav', '.npy'))
                save_features(audio_path, output_path, n_mfcc, max_length)
                files_processed += 1
    
    if files_processed == 0:
        print(f"No audio files found in {input_dir}")

if __name__ == '__main__':
    input_dir = 'data/raw_audios'
    output_dir = 'data/features/genre'
    process_directory(input_dir, output_dir)
