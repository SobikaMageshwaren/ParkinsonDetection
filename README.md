🎙️ Parkinson's Disease Audio Analysis Tool

This tool analyzes audio recordings to detect potential early signs of Parkinson's disease by extracting acoustic features and applying a pre-trained machine learning classifier.

🚀 Features

Audio Input: Record audio or upload a .wav file.

Pre-trained Model: Utilizes facebook/wav2vec2-base-960h for feature extraction.

Acoustic Feature Extraction: Jitter, Shimmer, Pitch, Harmonics-to-Noise Ratio (HNR).

Machine Learning Classifier: Supports SVM-based detection of early signs.

Diagnosis Output: Provides analysis with probabilities and early sign descriptions.

History Tracking: Saves analysis results to analysis_results/audio_analysis_results.csv.

Streamlit UI: User-friendly interface.

Command-Line Support: Terminal-based version included.

🛠️ Setup and Installation

1️⃣ Install Dependencies

Ensure Python 3.8+ is installed. Then install the required packages:

2️⃣ Download Pretrained Model

The script downloads the Wav2Vec2 model automatically:

3️⃣ Prepare SVM Model and Scaler

Ensure svm_model.pkl and scaler.pkl are available. Train your model beforehand and save it using:

🎯 Usage

▶️ Run the Streamlit App

🎧 Upload or Record Audio

Upload a .wav file.

Or click Record Audio (5 seconds).

🔍 View Results

Diagnosis: Early signs probability.

Acoustic Features: Jitter, Shimmer, Pitch, HNR.

Early Signs Description.

🗂 View Analysis History

Check saved analyses on the sidebar.

🖥️ Command Line Interface

For a terminal-based experience:

Press Enter to record.

View diagnosis, acoustic features, and analysis.

Type 'exit' to quit.

🧠 How It Works

1️⃣ Audio Recording/Upload

Records audio (5 seconds) or uploads .wav file.

2️⃣ Acoustic Feature Extraction

Extracts Jitter, Shimmer, Pitch, HNR using librosa.

3️⃣ Wav2Vec2 Embeddings

Processes audio into embeddings via Wav2Vec2.

4️⃣ SVM Prediction

Scales embeddings, predicts early signs probability.

5️⃣ Diagnosis Output

Returns diagnosis and early signs description.

6️⃣ Saves Results

Appends analysis data to CSV file.

📌 Example Diagnosis Output

🧩 Troubleshooting

Missing Model or Scaler Files

If svm_model.pkl or scaler.pkl is missing:

✅ Solution: Ensure the model files exist or retrain the classifier.

Feature Extraction Errors

✅ Solution: Ensure audio quality is good and has a valid .wav format.

📍 Future Enhancements

🎯 Support for other neurological disorder detection.

📈 Enhanced feature engineering.

🎛️ Hyperparameter tuning for better prediction accuracy.

📲 Mobile-friendly version.

💡 Acknowledgments

Facebook's Wav2Vec2 Model for state-of-the-art audio processing.

Librosa for acoustic feature extraction.

Streamlit for interactive UI.
