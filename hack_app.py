import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from pandas_profiling import ProfileReport

# Load the TF-IDF vectorizer and the model
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('model.pkl', 'rb') as file:
    clf = pickle.load(file)

st.title('FAKE NEWS CLASSIFIER')
abouts = st.sidebar.radio('Select Your Option', ('Text Classifier', 'Text Analysis' , 'Team'))

def classify_news():
    text_input = st.text_area('Enter the text to classify:', '')

    if st.button('Classify'):
        if text_input.strip() == '':
            st.error('Please enter some text.')
        else:
            text_tfidf = tfidf_vectorizer.transform([text_input])
            prediction = clf.predict(text_tfidf)

            if prediction[0] == 0:
                st.success('Prediction: Fake News')
            else:
                st.success('Prediction: True News')

def team():
    st.write("PRIYANSHI SHAH")
    st.write("DIYA RAMANI")
    st.write("KASHISH DHOKA")
    st.write("NYSA SINGH")
    st.write("PRASHANSA PAL")
    st.write("SHREYANSH SINGH")



if abouts == "Text Classifier":
    classify_news()
    
if abouts == "Team":
    team()


