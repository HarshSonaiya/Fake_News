from fastapi import FastAPI
from controllers.news_controller import predict_news

app = FastAPI()

@app.post("/predict")
async def predict(input: dict):
    print("main.py:",input)
    return await predict_news(input)
