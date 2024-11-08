import pickle
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the model
with open('66130701715sentiment_pipeline_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Streamlit app
st.title("Sentiment Analysis App")

# Input text
user_input = st.text_input("Enter a sentence:")

if user_input:
    # Make a prediction
    prediction = loaded_model.predict([user_input])[0]
    
    # Display the raw prediction output for debugging
    st.write("Raw Prediction Output:", prediction)
    
    # Display the result based on prediction output
    if prediction == 1:
        st.write("Sentiment: Positive")
    elif prediction == 0:
        st.write("Sentiment: Negative")
    else:
        st.write("Sentiment: Neutral or Unrecognized")
