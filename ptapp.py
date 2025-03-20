import streamlit as st
import sounddevice as sd
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import os
import wave

# Environment setup
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load pretrained models
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Load trained classifier and scaler
try:
    with open("svm_model.pkl", "rb") as model_file:
        classifier = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except (FileNotFoundError, pickle.UnpicklingError):
    st.error("ERROR: Missing or corrupted 'svm_model.pkl' and 'scaler.pkl'. Please train the model first.")
    st.stop()

# Setup output directory and results file
output_dir = "analysis_results"
os.makedirs(output_dir, exist_ok=True)
csv_file_path = os.path.join(output_dir, "audio_analysis_results.csv")
if not os.path.exists(csv_file_path):
    with open(csv_file_path, "w") as f:
        f.write("User Input Audio,Extracted Acoustic Features,Feature Analysis,Diagnosis,Early Signs Description\n")

# Function to record audio
def record_audio(duration=5, sr=16000):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio), sr

# Save audio as .wav file
def save_audio(audio, sr, file_name):
    file_path = os.path.join(output_dir, file_name)
    with wave.open(file_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return file_path

# Extract acoustic features
def extract_acoustic_features(audio, sr=16000):
    try:
        jitter = np.std(np.diff(np.where(librosa.zero_crossings(audio, pad=False))))
        shimmer = np.mean(librosa.feature.rms(y=audio))
        pitch = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        hnr = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
        return {"Jitter": jitter, "Shimmer": shimmer, "Pitch": pitch, "HNR": hnr}
    except Exception as e:
        st.error(f"Error extracting acoustic features: {e}")
        return {"Jitter": None, "Shimmer": None, "Pitch": None, "HNR": None}

# Detect early signs
def detect_early_signs(features):
    signs = []
    if features["Jitter"] > 0.02:
        signs.append("voice instability (jitter)")
    if features["Shimmer"] > 0.03:
        signs.append("vocal shakiness (shimmer)")
    if features["Pitch"] < 120:
        signs.append("softer or monotone voice")
    if features["HNR"] < 20:
        signs.append("breathy or weak voice")
    return "Early signs detected: " + ", ".join(signs) if signs else "No significant early signs detected."

# Analyze audio
def analyze_audio(audio, sr, audio_file_name):
    acoustic_features = extract_acoustic_features(audio, sr)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = wav2vec_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    scaled_embeddings = scaler.transform([embeddings])
    prediction = classifier.predict(scaled_embeddings)
    probability = classifier.predict_proba(scaled_embeddings)[0][1]

    diagnosis = "No early signs of Parkinson's disease detected."
    if prediction[0] == 1:
        diagnosis = f"Early signs of Parkinson's disease detected with a probability of {probability * 100:.2f}%."

    early_signs_description = detect_early_signs(acoustic_features)
    analysis_data = {
        "User Input Audio": audio_file_name,
        "Extracted Acoustic Features": acoustic_features,
        "Feature Analysis": early_signs_description,
        "Diagnosis": diagnosis,
        "Early Signs Description": early_signs_description
    }
    pd.DataFrame([analysis_data]).to_csv(csv_file_path, mode='a', header=False, index=False)
    return diagnosis, acoustic_features, early_signs_description

# Streamlit UI layout
st.title("🎙️ Parkinson's Disease Audio Analysis Tool")
st.markdown("### Upload or record audio to detect early signs of Parkinson's disease.")

# Align upload and record buttons on the same row
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
with col2:
    record_btn = st.button("Record Audio")

if record_btn:
    audio, sr = record_audio()
    audio_file_name = "user_recorded_audio.wav"
    save_audio(audio, sr, audio_file_name)
    st.success("✅ Recording complete!")
    uploaded_file = audio_file_name

if uploaded_file:
    st.audio(uploaded_file)
    st.info("Analyzing audio... Please wait.")
    audio, sr = librosa.load(uploaded_file, sr=16000)
    diagnosis, features, early_signs = analyze_audio(audio, sr, uploaded_file)

    st.subheader("📌 Diagnosis")
    st.write(diagnosis)
    st.subheader("📌 Acoustic Features")
    st.json(features)
    st.subheader("📌 Early Signs Description")
    st.write(early_signs)

st.markdown("---")

# Display analysis history
st.sidebar.title("📜 Analysis History")
if st.sidebar.button("Show Analysis History"):
    if os.path.exists(csv_file_path):
        history_df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        st.sidebar.dataframe(history_df)
    else:
        st.sidebar.warning("No analysis history available yet.")
def main():
    print("Welcome to the Parkinson's Disease Audio Analysis Tool!")
    while True:
        # Ask user for input to start recording
        user_choice = input("Press Enter to record audio or type 'exit' to quit: ").strip().lower()
        if user_choice == "exit":
            print("Exiting the program. Goodbye!")
            break

        # Record audio and save it
        audio, sr = record_audio(duration=5)
        audio_file_name = "user_input_audio.wav"
        save_audio(audio, sr, audio_file_name)

        # Analyze the audio and output the results
        print("\nAnalyzing audio... Please wait.")
        diagnosis, acoustic_features, feature_analysis, early_signs_description = analyze_audio(
            audio, sr, audio_file_name
        )

        # Display the results to the user
        print("\n✅ Analysis Complete!")
        print(f"📌 Diagnosis: {diagnosis}")
        print(f"📌 Acoustic Features: {acoustic_features}")
        print(f"📌 Feature Analysis: {feature_analysis}")
        print(f"📌 Early Signs Description: {early_signs_description}")
        print("-" * 50)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
