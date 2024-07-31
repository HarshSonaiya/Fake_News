import streamlit as st
import requests

# Streamlit application title
st.title('Fake News Classifier')

#Input field for news title 
news_title = st.text_area('Enter news title:')

#Input field for news author
news_author = st.text_area('Enter Author Name:')

# Input field for news text
news_text = st.text_area('Enter news text:')

user_input = {
    "title": news_title,
    "author": news_author,
    "text": news_text
}

if st.button('Classify'):
    if user_input:
        # Send POST request to FastAPI
        response = requests.post("http://fastapi:8000/predict", json=user_input)
        
        if response.status_code == 200:
            result = response.json()
            st.write(f"Prediction: {result['prediction']}")
        else:
            st.write("Error: Could not get prediction.")
    else:
        st.write("Please enter some text.")
