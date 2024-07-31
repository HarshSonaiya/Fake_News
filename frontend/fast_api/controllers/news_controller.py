import requests

BENTO_SERVICE_URL = "http://localhost:3000/predict"

def predict_news(input: dict):

    print("news_controller:",input,type(input))
    response = requests.post(BENTO_SERVICE_URL, json={"input":input})
    print("news_controller:",response.json())
    return response.json()
