import joblib
import pandas as pd

# Load model
model = joblib.load("models/phishing_model.pkl")

def detect_phishing(url):
    # Extract features
    features = {
        'url_length': len(url),
        'has_https': 1 if 'https' in url else 0
    }
    features_df = pd.DataFrame([features])

    # Predict
    prediction = model.predict(features_df)
    return "Phishing" if prediction[0] == 1 else "Legitimate"

# Example usage
url = "https://example.com/login"
result = detect_phishing(url)
print(f"The URL '{url}' is classified as: {result}")
