import joblib
import re

# Load trained model & vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/spam_vectorizer.pkl")

# cleaning function 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

test_comments = [
    "Amazing explanation, very helpful!",
    "Subscribe to my channel for free giveaway",
    "This video is misleading and fake",
    "Click this link to earn money fast",
    "Good content but audio quality is poor",

    # Tricky spam starts here
    "Loved the video! I also shared something similar on my channel 😊",
    "Great tutorial bro, check my profile for the full method",
    "This actually worked for me, details in my bio",
    "Thanks! I made a video solving the same problem step-by-step",
    "Nice explanation, people are earning daily from this link",
    "Who else is watching in 2026? I dropped a useful link",
    "This changed my life, message me for proof",
    "Honestly helpful 👍 subscribe if you want more tricks",
    "I was confused before, now earning online easily",
    "For those asking, the solution is in my profile"
]


# Clean comments
cleaned_comments = [clean_text(c) for c in test_comments]

# Vectorize
X_test = vectorizer.transform(cleaned_comments)

# Predict
predictions = model.predict(X_test)

# Display results
for comment, pred in zip(test_comments, predictions):
    label = "SPAM ❌" if pred == 1 else "NOT SPAM ✅"
    print(f"\nComment: {comment}")
    print(f"Prediction: {label}")
