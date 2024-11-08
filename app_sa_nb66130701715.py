import pickle
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pythainlp import word_tokenize
import re

def preprocess_text(text):
  """ทำความสะอาดข้อความ"""
  text = text.lower()  # เปลี่ยนเป็นตัวพิมพ์เล็ก
  text = re.sub(r'[^\w\s]', '', text)  # ลบสัญลักษณ์
  tokens = word_tokenize(text, engine="newmm")
  stopwords = ["และ", "หรือ", "เป็นต้น"]  # คำหยุดที่กำหนดเอง
  tokens = [w for w in tokens if w not in stopwords]
  return " ".join(tokens)

# Reload the model or create it if retraining
try:
  # Load the model
  with open('sentiment_pipeline_model.pkl', 'rb') as model_file:
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
  with open('sentiment_pipeline_model.pkl', 'wb') as model_file:
    pickle.dump(loaded_model, model_file)

# Streamlit app
st.title("Sentiment Analysis App (Thai Support)")

# Input text
user_input = st.text_input("กรอกข้อความที่ต้องการทำนาย:")

if user_input:
  # Preprocess the input
  user_input = preprocess_text(user_input)

  # Make a prediction
  prediction = loaded_model.predict([user_input])[0]

  # Display the result based on prediction output
  if prediction == 1:
    st.write("Sentiment: Positive (บวก)")
  else:
    st.write("Sentiment: Negative (ลบ)")
