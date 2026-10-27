
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message to check whether it is Spam or Ham.")

model_path = Path(__file__).parent / "spam_classifier.joblib"

if not model_path.exists():
    st.error("Trained model not found. Please train the model first.")
    st.stop()

model = joblib.load(model_path)

message = st.text_area("Enter your SMS message:")

if st.button("Check Message"):
    if message.strip():
        spam_probability = model.predict_proba([message])[0][
            list(model.classes_).index("spam")
]
        if st.metric("Estimated Spam Score", f"{spam_probability:.1%}"):
           st.progress(float(spam_probability))
        else:
            st.success("Result: HAM ✅")
    else:
        st.warning("Please enter a message first.")
