import httpx
from pydantic import BaseModel

class News(BaseModel):
    text: str

BENTO_SERVICE_URL = "http://bentoml:5000/predict"

async def predict_news(input: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(BENTO_SERVICE_URL, json=input)
        return response.json()
