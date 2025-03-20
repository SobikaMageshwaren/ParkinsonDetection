import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Example placeholder data (replace with actual Wav2Vec2 embeddings and labels)
X = np.random.rand(100, 768)  # Replace with actual Wav2Vec2 feature embeddings
y = np.random.choice([0, 1], size=(100,))  # Labels: 0 (Healthy), 1 (Parkinson's)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM Classifier
classifier = SVC(kernel="linear", probability=True)
classifier.fit(X_train, y_train)

# Evaluate the Model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the Model and Scaler
with open("svm_model.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("SVM model and scaler saved as 'svm_model.pkl' and 'scaler.pkl'.")
