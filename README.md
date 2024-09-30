# Audio Genre Classifier

**Audio Genre Classifier** is an AI-based tool designed to classify audio clips into the following genres:
- **Classical**
- **Jazz**
- **Rock**

## Features
- Extracts MFCC (Mel-frequency cepstral coefficients) features from audio files.
- Uses a deep learning model to predict the genre of audio clips.
- Two ways to run the classification:
  1. Via a standalone script (`predict.py`).
  2. Via a FastAPI app (`app.py`).

## Running the Project in Order

To fully run the project, follow these steps in order:

1. **Extract MFCC features** from the audio files and store them in the `features/` folder:
   ```bash
   python models/extract_features.py
   ```

2. **Train the model** on the audio files in the `data/raw_audios/` folder:
   ```bash
   python models/train.py
   ```

3. **Predict the genre** using either of the two methods described below:
   - Via the standalone script `predict.py`
   - Via the FastAPI app `app.py`

## How to Run

### 1. Running Audio Classification using `predict.py`

This process allows you to classify audio files by placing them into the `Sample/` folder and running the standalone script.

#### Steps:
1. Place your audio files (`.mp3`, `.wav`) into the `Sample/` folder.
2. Run the following command to classify the audio files:
   ```bash
   python models/predict.py
   ```
   The script will extract MFCC features from each audio file and predict the genre.

### 2. Running Audio Classification through FastAPI (`app.py`)

You can run a FastAPI app to classify audio by uploading them via an HTTP POST request.

#### Steps:
1. Install FastAPI and Uvicorn:
   ```bash
   pip install fastapi uvicorn
   ```
2. Start the FastAPI app:
   ```bash
   uvicorn app:app --reload
   ```
3. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to access the interactive API documentation.
4. Send a POST request to the `/predict/` endpoint with an audio file to classify. Here’s an example using `curl`:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@path_to_your_audio_file"
   ```
   Alternatively, you can use tools like **Postman** to upload files and receive predictions.

### Testing with Example Audio Files

For testing the classifier, you can use the training audio files provided in the `data/raw_audios/` folder. These example audio files are good for basic testing, but you can add more files to improve the model's accuracy.

## Project Structure

```
Audio_Classifier/
├── data/                               # Contains raw audio files and extracted features
│   ├── raw_audios/                     # Folder to store raw audio files for training
│   │   └── genre/                      # Subfolders for different genres of audio
│   │       ├── classical/              # Raw classical audio files
│   │       ├── jazz/                   # Raw jazz audio files
│   │       └── rock/                   # Raw rock audio files
│   └── features/                       # Folder to store extracted MFCC features for training
│       └── genre/                      # Subfolders for different genres of extracted features
│           ├── classical/              # Extracted features for classical genre
│           ├── jazz/                   # Extracted features for jazz genre
│           └── rock/                   # Extracted features for rock genre
├── models/                             # Contains all model-related scripts
│   ├── extract_features.py             # Script to extract MFCC features from audio files
│   ├── model.py                        # Defines the deep learning model architecture
│   ├── train.py                        # Script to train the model on extracted features
│   ├── predict.py                      # Script to predict audio genre using the trained model
│   └── model_weights.h5                # Pre-trained model weights
├── audio_app/                          # Folder to store uploaded audio files when using FastAPI app
│   └── (empty folder to store uploaded audio files)
├── Sample/                             # Folder to store audio files for running predict.py
│   └── (empty folder to store audio files to run predict.py)
├── app.py                              # FastAPI app to serve the audio classifier via HTTP requests
├── requirements.txt                    # List of dependencies required for the project
├── README.md                           # Documentation on how to use the Audio Classifier
└── .venv/                              # Virtual environment for package management (not tracked in Git)

```

### Folder Descriptions:
- **data/raw_audios/**: Stores raw audio files organized into folders by genre.
- **data/features/**: Stores extracted MFCC features for training and testing.
- **models/extract_features.py**: Script that extracts MFCC features from audio files and saves them for training.
- **models/**: Contains the model architecture, training, and prediction scripts.
- **audio_app/**: Stores uploaded audio files for prediction through the FastAPI app.

## Requirements

Ensure the following packages are installed:

```
tensorflow
keras
numpy
librosa
fastapi
uvicorn
```

## License
This project is licensed under the MIT License.