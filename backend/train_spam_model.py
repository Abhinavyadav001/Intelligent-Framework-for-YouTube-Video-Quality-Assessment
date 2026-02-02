import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("datasets/youtube_spam.csv")

# -----------------------------
# 2. Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_content"] = df["CONTENT"].apply(clean_text)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X = df["clean_content"]
y = df["CLASS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. Word + Character TF-IDF
# -----------------------------
word_tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3
)

vectorizer = FeatureUnion([
    ("word_tfidf", word_tfidf),
    ("char_tfidf", char_tfidf)
])

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5. Hyperparameter Tuning (SVM)
# -----------------------------
param_grid = {
    "C": [0.1, 0.5, 1, 2, 3, 5, 10]
}


svm = LinearSVC(
    class_weight="balanced",
    dual="auto"
)

grid = GridSearchCV(
    svm,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train_vec, y_train)

best_model = grid.best_estimator_

print("Best C:", grid.best_params_)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = best_model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 7. Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/spam_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")
