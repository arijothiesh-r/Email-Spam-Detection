```python
import streamlit as st
import pickle
import string
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# ---------------------------------------
# NLTK
# ---------------------------------------

nltk.download("punkt")
nltk.download("stopwords")


# ---------------------------------------
# Stemmer
# ---------------------------------------

ps = PorterStemmer()


# ---------------------------------------
# Text Transformation Function
# ---------------------------------------

def transform_text(text):

    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric words
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# ---------------------------------------
# Get current folder
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------
# File paths
# ---------------------------------------

vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")


# ---------------------------------------
# Check files
# ---------------------------------------

if not os.path.exists(vectorizer_path):

    st.error("❌ vectorizer.pkl not found!")

    st.info(
        "Put vectorizer.pkl inside the same folder as app.py."
    )

    st.stop()


if not os.path.exists(model_path):

    st.error("❌ model.pkl not found!")

    st.info(
        "Put model.pkl inside the same folder as app.py."
    )

    st.stop()


# ---------------------------------------
# Load Vectorizer
# ---------------------------------------

with open(vectorizer_path, "rb") as file:
    tfidf = pickle.load(file)


# ---------------------------------------
# Load Model
# ---------------------------------------

with open(model_path, "rb") as file:
    model = pickle.load(file)


# ---------------------------------------
# Streamlit UI
# ---------------------------------------

st.title("📧 Email/SMS Spam Classifier")

st.write(
    "Enter an Email or SMS message to check whether it is Spam or Not Spam."
)


# ---------------------------------------
# Input
# ---------------------------------------

input_sms = st.text_input(
    "Enter the message"
)


# ---------------------------------------
# Prediction
# ---------------------------------------

if st.button("Predict"):

    if input_sms.strip() == "":
        
        st.warning("⚠️ Please enter a message.")

    else:

        # Transform text
        transform_sms = transform_text(input_sms)

        # Convert text into TF-IDF
        vector_input = tfidf.transform([transform_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:

            st.error("🚨 Spam")

        else:

            st.success("✅ Not Spam")
```
