
# SMS Spam Classifier

A Machine Learning web app that classifies SMS messages as Spam or Ham.

## Features
- SMS spam classification
- TF-IDF text feature extraction
- Multinomial Naive Bayes classifier
- Hyperparameter tuning with GridSearchCV
- Spam probability score
- Interactive Streamlit interface

## Dataset
SMS Spam Collection dataset from the UCI Machine Learning Repository.

## Technologies
- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Run Locally

1. Install dependencies:

   `python -m pip install -r requirements.txt`

2. Train the model:

   `python train.py`

3. Launch the web app:

   `python -m streamlit run app.py`

## Evaluation

Previously recorded test results:
- Accuracy: 98.07%
- Spam Precision: 0.96
- Spam Recall: 0.89
- Spam F1-score: 0.92

Note: These metrics describe the earlier evaluated model and should be reverified after retraining.

## Project Files

- `train.py` — trains and evaluates the model
- `predict.py` — predicts Spam or Ham from an SMS
- `app.py` — Streamlit web application
- `requirements.txt` — Python dependencies
- `SMSSpamCollection` — dataset file
- `spam_classifier.joblib` — trained model artifact
