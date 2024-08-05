import httpx
from pydantic import BaseModel

class News(BaseModel):
    text: str

BENTO_SERVICE_URL = "http://172.17.0.2:3000/predict"
# BENTO_FILE_SERVICE_URL = "http://172.17.0.3:3000/predict_file"

async def predict_news(input: dict):
    async with httpx.AsyncClient() as client:
        print("NC.py:",input)
        response = await client.post(BENTO_SERVICE_URL, json={"input":input})
        return response.json()

# async def predict_file_from_bento_service(file_content: bytes) -> bytes:
#     async with httpx.AsyncClient() as client:
#         response = await client.post(BENTO_FILE_SERVICE_URL, content=file_content)
#         response.raise_for_status()
#         return response.content

