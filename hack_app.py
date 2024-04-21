import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
from transformers import pipeline

# Load the TF-IDF vectorizer and the SVC model for fake tweet classification
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('svm_classifier.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

# Load the TF-IDF vectorizer and the model for fake news classification
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer_news = pickle.load(file)

with open('model.pkl', 'rb') as file:
    clf = pickle.load(file)

# summarizer = pipeline("summarization")

st.title('Fake News Classifier')
abouts = st.sidebar.radio('Select Your Option', ('Text Classifier', 'Text Analysis' , 'Fake Tweet Classifier', 'Team'))

def classify_news():
    st.subheader("News Classifcation")
    text_input = st.text_area('Enter the text to classify:', '')

    if st.button('Classify'):
        if text_input.strip() == '':
            st.error('Please enter some text.')
        else:
            text_tfidf = tfidf_vectorizer_news.transform([text_input])
            prediction = clf.predict(text_tfidf)

            if prediction[0] == 0:
                st.success('Prediction: Fake News')
            else:
                st.success('Prediction: True News')
                # summary = summarizer(text_input, max_length=150, min_length=30, do_sample=False)
                # st.write('Summary of True News:')
                # st.write(summary[0]['summary_text'])

def classify_tweet():
    st.subheader("Tweet Classification")
    user_tweet = st.text_input('Enter the tweet:')

    if st.button('Predict'):
        if user_tweet.strip() == '':
            st.warning('Please enter a tweet.')
        else:
            prediction = predict_fake_tweet(user_tweet)
            st.success(f'Prediction: {prediction}')

def predict_fake_tweet(tweet):
    tweet_vectorized = tfidf_vectorizer.transform([tweet])
    prediction = svm_classifier.predict(tweet_vectorized)
    return "Fake" if prediction == 1 else "Real"

def team():
    st.subheader("Team")
    st.write("PRIYANSHI SHAH  - 21BIT0378 ")
    st.write("DIYA RAMANI  - 21BIT0392")
    st.write("KASHISH DHOKA  - 21BIT0021")
    st.write("NYSA SINGH  - 21BIT0376")
    st.write("PRASHANSA PAL  - 21BIT0231")
    st.write("SHREYANSH SINGH  - 21BIT0604")

if abouts == "Text Classifier":
    classify_news()

if abouts == "Fake Tweet Classifier":
    classify_tweet()

if abouts == "Team":
    team()
