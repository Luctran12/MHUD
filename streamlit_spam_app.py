import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ===================== CONFIG =====================
st.set_page_config(page_title="Spam Email Detector", layout="centered")

# ===================== CACHE MODEL =====================
@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===================== CACHE NLTK =====================
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

stopwords_set = load_nltk()

# ===================== CACHE PREPROCESS =====================
@st.cache_resource
def load_preprocess():
    stemmer = PorterStemmer()
    return stemmer

stemmer = load_preprocess()

# ===================== PREPROCESS FUNCTION =====================
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(words)

# ===================== UI =====================
st.title("📧 Spam Email Detection App")
st.write("Upload or type an email to check whether it is spam or not.")

# Input box
user_input = st.text_area("✍️ Enter email content:", height=200)

# Predict button
if st.button("🔍 Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocess
        processed = preprocess_text(user_input)

        # Vectorize
        X_input = vectorizer.transform([processed])

        # Predict
        proba = model.predict_proba(X_input)[0]
        spam_prob = proba[1]

        # Display result
        st.subheader("📊 Result")
        st.write(f"Spam probability: **{spam_prob:.4f}**")

        if spam_prob > 0.9:
            st.error("🚫 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM")

# ===================== EXTRA =====================
st.markdown("---")
st.markdown("### 💡 Tips:")
st.markdown("- Spam emails often contain promotional or suspicious content.")
st.markdown("- Normal emails include work, study, or personal communication.")