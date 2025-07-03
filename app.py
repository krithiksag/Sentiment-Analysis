import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources silently
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load the pre-trained classifier
with open("naive_bayes_model.pkl", "rb") as f:
    classifier = pickle.load(f)

# Feature extraction function
def extract_features(text):
    tokens = word_tokenize(text.lower())
    words = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')]
    return {word: True for word in words}

# Streamlit page settings
import streamlit as st

st.set_page_config(page_title="Sentiment Classifier", layout="centered")

# Inject CSS
st.markdown("""
    <style>
        /* Ensure the title class is uniquely scoped */
        .big-title {
            font-size: 70px !important;
            color: white !important;
            text-align: center !important;
            margin-top: 20px !important;
            margin-bottom: 30px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Render the title
st.markdown('<h1 class="big-title">Sentiment Classifier</h1>', unsafe_allow_html=True)


# Description
st.write("Enter a sentence below to determine whether the sentiment is **positive** or **negative**.")

# Text input
user_input = st.text_area("Your Sentence", placeholder="Example: I really enjoyed the movie.")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip():
        features = extract_features(user_input)
        prediction = classifier.classify(features)

        if prediction.lower() == "pos":
            st.success(f"Predicted Sentiment: {prediction.upper()} (Positive)")
        elif prediction.lower() == "neg":
            st.error(f"Predicted Sentiment: {prediction.upper()} (Negative)")
        else:
            st.info(f"Predicted Sentiment: {prediction.upper()}")
    else:
        st.warning("Please enter some text before clicking predict.")

# Footer
st.markdown('<p class="footer">Built using  Naive Bayes Model</p>', unsafe_allow_html=True)
