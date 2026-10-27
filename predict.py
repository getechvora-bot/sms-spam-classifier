
import os
import joblib

# Saved model ka path
MODEL_PATH = "spam_classifier.joblib"

# Check karo ke model file mojood hai
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file nahi mili. Pehle python train.py run karo."
    )

# Trained model load karo
model = joblib.load(MODEL_PATH)

def predict_message(message):
    """Message ko spam ya ham classify karo."""
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    # Model ke classes ke mutabiq probability nikalo
    spam_index = list(model.classes_).index("spam")
    spam_probability = probabilities[spam_index]

    return prediction, spam_probability


# User se message lo
if __name__ == "__main__":
    message = input("Apna SMS message enter karo: ")

    prediction, spam_probability = predict_message(message)

    print("\nPrediction:", prediction)
    print(f"Spam probability: {spam_probability:.2%}")
