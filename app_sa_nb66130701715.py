import pickle
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from pythainlp import word_tokenize

# Custom tokenizer for Thai
def thai_tokenizer(text):
    return word_tokenize(text, engine="newmm")

# Reload the model or create it if retraining
try:
    # Load the model
    with open('66130701715sentiment_pipeline_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except FileNotFoundError:
    # If no pre-trained model exists, create a new pipeline for training
   loaded_model = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, ngram_range=(1, 2))),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))  # เปลี่ยนเป็น Random Forest
  ])
    # Here, you would train the model with your Thai dataset
    # Example:
    # loaded_model.fit(train_data_thai, train_labels_thai)
    # Then save it for future use
    with open('66130701715sentiment_pipeline_model.pkl', 'wb') as model_file:
        pickle.dump(loaded_model, model_file)

# Streamlit app
st.title("Sentiment Analysis App (Thai Support)")

# Input text
user_input = st.text_input("กรอกข้อความที่ต้องการทำนาย:")

if user_input:
    # Make a prediction
    prediction = loaded_model.predict([user_input])[0]
    
    # Display the raw prediction output for debugging
    st.write("Raw Prediction Output:", prediction)
    
    # Display the result based on prediction output
    if prediction == 1:
        st.write("Sentiment: Positive (บวก)")
    elif prediction == 0:
        st.write("Sentiment: Negative (ลบ)")
    else:
        st.write("Sentiment: Neutral or Unrecognized (เป็นกลางหรือไม่รู้จัก)")
