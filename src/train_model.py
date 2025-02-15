# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
phishing_data = pd.read_csv("data/phishing_urls.csv")
legitimate_data = pd.read_csv("data/legitimate_urls.csv")

# Combine datasets
data = pd.concat([phishing_data, legitimate_data])
data['label'] = data['label'].map({'phishing': 1, 'legitimate': 0})

# Feature extraction (example features)
data['url_length'] = data['url'].apply(len)
data['has_https'] = data['url'].apply(lambda x: 1 if 'https' in x else 0)

# Split data
X = data[['url_length', 'has_https']]  # Add more features as needed
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save model
joblib.dump(model, "models/phishing_model.pkl")
