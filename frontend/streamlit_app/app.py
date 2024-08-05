import streamlit as st
import requests
import pandas as pd


# Streamlit application title
st.title('Fake News Classifier')

# Option to classify a single news article or upload a CSV file
option = st.selectbox(
    'Choose how you want to classify news:',
    ('Single News Article', 'Upload CSV File')
)

# Handling single news article classification
if option == 'Single News Article':
    # Input field for news title 
    news_title = st.text_input('Enter news title:')

    # Input field for news author
    news_author = st.text_input('Enter Author Name:')

    # Input field for news text
    news_text = st.text_area('Enter news text:')

    user_input = {
        "title": news_title,
        "author": news_author,
        "text": news_text
    }

    if st.button('Classify'):
        if user_input['title'] and user_input['author'] and user_input['text']:
    
            # Send POST request to FastAPI
            response = requests.post("http://fastapi:8000/predict", json=user_input)

            if response.status_code == 200:
                result = response.json()
                st.write(f"Prediction: {result}")
            else:
                st.write("Error: Could not get prediction.")
        else:
            st.write("Please enter the title, author name, and text.")

# Handling CSV file upload for batch classification
elif option == 'Upload CSV File':
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        st.write(df)

        # Convert DataFrame to CSV string
        csv_data = df.to_csv(index=False)

        # Send POST request to FastAPI with CSV data
        response = requests.post("http://fastapi:8000/predict_file", files={"file": ("data.csv", csv_data)})

        if response.status_code == 200:
            result = response.json()
            result_df = pd.DataFrame(result)

            # Provide download option
            st.download_button(label="Download results", data=result_df.to_csv(index=False), file_name="result.csv", mime="text/csv")
        else:
            st.write("Error: Could not get prediction results.")