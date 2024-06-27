"""
Phish-Scan: A Python-based phishing link scanner
Author: Muhammed Farhan
"""

import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlparse
from colorama import init, Fore

# Initialize colorama
init(autoreset=True)

# Sample data for training
phishing_urls = ["http://phishingsite.com", "http://malicious.com", "http://fraudsite.net"]
legitimate_urls = ["http://google.com", "http://github.com", "http://example.com"]

# Labels: 1 for phishing, 0 for legitimate
labels = [1, 1, 1, 0, 0, 0]

# Combine data
urls = phishing_urls + legitimate_urls

# Vectorize URLs
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(urls)

# Train the model
model = LogisticRegression()
model.fit(X, labels)

# Save the model and vectorizer
joblib.dump(model, 'phish_scan_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load pre-trained model
def load_model():
    try:
        model = joblib.load('phish_scan_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
    except FileNotFoundError:
        model = None
        vectorizer = None
    return model, vectorizer

# Simple heuristic-based checks
def heuristic_checks(url):
    # Known phishing domains (for demo purposes, use a more comprehensive list)
    known_phishing_domains = ["phishingsite.com", "malicious.com", "fraudsite.net"]
    domain = urlparse(url).netloc
    if domain in known_phishing_domains:
        return True
    return False

# Use a machine learning model to predict phishing
def ml_checks(url, model, vectorizer):
    if model and vectorizer:
        features = vectorizer.transform([url])
        prediction = model.predict(features)
        return prediction[0] == 1
    return False

# Main function to scan URL
def scan_url(url):
    # Load the model and vectorizer
    model, vectorizer = load_model()

    # Perform heuristic checks
    if heuristic_checks(url):
        return Fore.RED + "This URL is potentially a phishing link."

    # Perform machine learning checks
    if ml_checks(url, model, vectorizer):
        return Fore.RED + "This URL is potentially a phishing link."

    return Fore.GREEN + "This URL appears to be safe."

if __name__ == "__main__":
    # URL to check
    url = input("Enter the URL to check: ")
    result = scan_url(url)
    print(result)
