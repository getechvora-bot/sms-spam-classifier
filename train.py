
import os
import urllib.request
import zipfile
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)

# 1. Dataset download karo agar file mojood nahi
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_name = "sms_spam_collection.zip"
file_name = "SMSSpamCollection"

if not os.path.exists(file_name):
    print("Downloading dataset...")

    urllib.request.urlretrieve(url, zip_name)

    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(".")

    print("Dataset ready.")

# 2. Dataset load aur clean karo
df = pd.read_csv(
    file_name,
    sep="\t",
    header=None,
    names=["label", "message"]
)

df = df.drop_duplicates().reset_index(drop=True)

print("Cleaned dataset shape:", df.shape)
print("\nLabel counts:")
print(df["label"].value_counts())

# 3. Features aur target alag karo
X = df["message"]
y = df["label"]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# 5. Pipeline banao
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# 6. Cross-validation aur scoring setup
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

scoring = {
    "precision_spam": make_scorer(
        precision_score,
        pos_label="spam",
        zero_division=0
    ),
    "recall_spam": make_scorer(
        recall_score,
        pos_label="spam",
        zero_division=0
    ),
    "f1_spam": make_scorer(
        f1_score,
        pos_label="spam",
        zero_division=0
    ),
    "accuracy": "accuracy"
}

# 7. Alpha tune karo
param_grid = {
    "classifier__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit="f1_spam",
    cv=cv,
    n_jobs=-1
)

print("\nTraining and tuning model...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters:", grid_search.best_params_)
print("Best CV spam F1:", round(grid_search.best_score_, 3))

# 8. Test set par final evaluation
y_pred = best_model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nConfusion Matrix (ham, spam):")
print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=["ham", "spam"],
    zero_division=0
))

# 9. Trained model save karo
model_path = "spam_classifier.joblib"
joblib.dump(best_model, model_path)

print(f"\nModel saved successfully: {model_path}")
