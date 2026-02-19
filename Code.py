import pandas as pd
import numpy as np
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text


# -------------------------------
# Load Dataset
# -------------------------------
def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df


# -------------------------------
# Train Model
# -------------------------------
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


# -------------------------------
# Save Model
# -------------------------------
def save_model(model, vectorizer):
    with open("spam_model.pkl", "wb") as f:
        pickle.dump((model, vectorizer), f)


# -------------------------------
# Load Model
# -------------------------------
def load_model():
    with open("spam_model.pkl", "rb") as f:
        return pickle.load(f)


# -------------------------------
# Predict Function
# -------------------------------
def predict_email(text, model, vectorizer):
    text = clean_text(text)
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return "Spam" if prediction == 1 else "Not Spam"


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    df = load_data("dataset/spam.csv")
    model, vectorizer = train_model(df)
    save_model(model, vectorizer)

    # Example Prediction
    sample_email = "Congratulations! You've won a free lottery ticket."
    print("\nPrediction:", predict_email(sample_email, model, vectorizer))
