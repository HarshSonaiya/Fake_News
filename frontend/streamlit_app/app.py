import streamlit as st
import requests

# Streamlit application title
st.title('Fake News Classifier')

# Input field for news text
news_text = st.text_area('Enter news text:')

if st.button('Classify'):
    if news_text:
        # Send POST request to FastAPI
        response = requests.post("http://fastapi:8000/predict", json={"text": news_text})
        
        if response.status_code == 200:
            result = response.json()
            st.write(f"Prediction: {result['prediction']}")
        else:
            st.write("Error: Could not get prediction.")
    else:
        st.write("Please enter some text.")
