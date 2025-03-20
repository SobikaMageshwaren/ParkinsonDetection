import streamlit as st
import pandas as pd
import sounddevice as sd
import librosa
import numpy as np
import os
import soundfile as sf
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Define paths
metadata_path = "D:/6 SEM LAB/Parkison/dataset/metadata.xlsx"
history_file = "audio_recordings_history.csv"

# Load metadata
metadata = pd.read_excel(metadata_path)

# Load pre-trained model
if os.path.exists("parkinson_disease_model.pkl"):
    model = joblib.load("parkinson_disease_model.pkl")
else:
    st.error("No pre-trained model found. Please train the model first.")
    st.stop()

# Feature extraction function with named features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if sr < 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        n_fft = max(256, min(512, len(y)))
        hop_length = n_fft // 2

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=n_fft, hop_length=hop_length)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=n_fft, hop_length=hop_length)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=200.0, n_bands=3, n_fft=n_fft, hop_length=hop_length)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz)
        ])

        feature_names = (
            [f"mfcc_{i}" for i in range(13)] +
            [f"chroma_{i}" for i in range(12)] +
            [f"mel_{i}" for i in range(40)] +
            [f"contrast_{i}" for i in range(3)] +
            [f"tonnetz_{i}" for i in range(6)]
        )

        return dict(zip(feature_names, features))

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Detect early signs function
def detect_early_signs(features):
    try:
        jitter = features.get("mfcc_0", 0)
        shimmer = features.get("mfcc_1", 0)
        tonnetz_variation = features.get("tonnetz_5", 0)

        reasons = []
        if jitter > 0.02:
            reasons.append("High Jitter")
        if shimmer > 0.1:
            reasons.append("High Shimmer")
        if tonnetz_variation > 0.5:
            reasons.append("Abnormal Tonnetz Variation")
        if np.mean([features.get(f"mfcc_{i}", 0) for i in range(13)]) < -4:
            reasons.append("Low MFCC Energy Levels")

        return ("Early signs detected" if reasons else "No early signs detected", reasons)
    
    except Exception as e:
        st.error(f"Error analyzing early signs: {e}")
        return "Error", []

# Audio recording function
def record_audio(duration=10, sampling_rate=16000):
    st.info(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1)
    sd.wait()
    audio_filename = f"recorded_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    sf.write(audio_filename, audio_data, sampling_rate)
    return audio_filename

# Prediction function
def predict_parkinson(audio_filename):
    features = extract_features(audio_filename)
    if features is None:
        st.error("Failed to extract features.")
        return

    feature_values = list(features.values())
    prediction_proba = model.predict_proba([feature_values])[0][1]
    prediction = "Healthy" if prediction_proba < 0.5 else "Parkinson's Disease"

    st.success(f"Prediction: {prediction}")
    st.info(f"Confidence: {prediction_proba * 100:.2f}%")

    # Ensure early_signs and reasons are always defined
    early_signs = "No early signs detected"
    reasons = []

    if prediction == "Parkinson's Disease":
        early_signs, reasons = detect_early_signs(features)
        st.info(f"Early signs analysis: {early_signs}")
        if reasons:
            st.warning(f"Possible reasons: {', '.join(reasons)}")

    feature_str = ", ".join([f"{name}: {value:.2f}" for name, value in features.items()])
    history_entry = f'{audio_filename},"{feature_str}","{prediction}",{prediction_proba:.2f},"{early_signs}","{"; ".join(reasons)}"\n'
    with open(history_file, "a") as history:
        history.write(history_entry)


# Display history function
def display_history():
    if os.path.exists(history_file):
        try:
            history_df = pd.read_csv(
                history_file,
                names=["Filename", "Features", "Prediction", "Probability", "Early Signs", "Reasons"],
                on_bad_lines='skip'
            )
            st.subheader("📜 Previous Analyses")
            st.dataframe(history_df)
        except Exception as e:
            st.error(f"Failed to load history: {e}")
    else:
        st.info("No previous records found.")

# Streamlit UI
st.title("🎙️ Parkinson's Disease Detection App")
st.markdown("### Upload an audio file or record your voice to analyze for early signs of Parkinson's Disease.")

input_mode = st.radio("Choose input mode:", ["Record Audio", "Upload Audio File"])

if input_mode == "Record Audio":
    if st.button("🎧 Start Recording"):
        audio_filename = record_audio()
        predict_parkinson(audio_filename)

elif input_mode == "Upload Audio File":
    uploaded_file = st.file_uploader("📁 Upload an audio file", type=["wav"])
    if uploaded_file is not None:
        audio_filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with open(audio_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        predict_parkinson(audio_filename)

# Display the history
display_history()