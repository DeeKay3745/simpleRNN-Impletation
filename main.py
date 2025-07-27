#step1: import all libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load the word index from IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

#load the pre-trained model with ReLU activation
model = load_model('simpleRNN_imdb.h5')

 #step2:Helper Functions

#function to decode reviews
def decode_rewiew(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


#STep:3 predictionfunction
def predict_sentiment(review):
    preprocess_input=preprocess_text(review)
    pred =model.predict(preprocess_input)
    sentiment = "+ve" if pred[0][0]>0.5 else "-ve"
    return sentiment, pred[0][0]

#step 4:User Input and prediction /. stremlit app

import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

#User input
user_input = st.text_area('Movie Review')
if st.button('Classify'):
    #example="This movie was fantastic! The acting was great and the plot was thrilling."
    sentimentt, prob = predict_sentiment(user_input)
    st.write(f"Review:{user_input}")
    st.write(f"Sentiment:{sentimentt}")
    st.write(f"Prediction score:{prob}")
else:
    st.write("Enter the review")
