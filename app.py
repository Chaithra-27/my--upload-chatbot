import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("faq.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["Question"])

st.title("FAQ Chatbot")

user_question = st.text_input("Ask your question")

if st.button("Get Answer"):
    user_vector = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, X)
    index = similarity.argmax()
    st.success(data["Answer"][index])