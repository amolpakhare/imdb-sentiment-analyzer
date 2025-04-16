import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# nltk.download('stopwords')
# nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load pre-trained models
cv = pickle.load(open('vectorizer (1).pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 10px;
    }
    .subheader {
        text-align: center;
        font-size: 20px;
        color: gray;
        margin-bottom: 40px;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-title">🎬 IMDB Movie Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter a movie review to predict if the sentiment is positive or negative.</div>', unsafe_allow_html=True)

# Input field
input_sms = st.text_input("Enter your review:")

# Predict button
if st.button('Predict Sentiment'):
    # Preprocess the input
    transformed_sms = transform_text(input_sms)

    # Vectorize the preprocessed text
    vector_input = cv.transform([transformed_sms])

    # Make prediction
    result = model.predict(vector_input)[0]

    # Display result with colored header
    if result == 1:
        st.success("✅ The sentiment is **Positive**!")
    else:
        st.error("❌ The sentiment is **Negative**.")

