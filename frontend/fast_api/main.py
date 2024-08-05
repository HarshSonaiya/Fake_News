from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from io import StringIO
import httpx
import pandas as pd
from controllers.news_controller import predict_news

app = FastAPI()

BENTO_FILE_SERVICE_URL = "http://172.17.0.2:3000/predict"

@app.post("/predict")
async def predict(input: dict):
    return await predict_news(input)

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    try:
        # Read the CSV file into DataFrame
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Convert DataFrame to JSON
        data = df.to_dict(orient='records')
        results = []
        # Send data to BentoML service
        async with httpx.AsyncClient() as client:

            for record in data:
                response = await client.post(BENTO_FILE_SERVICE_URL, json={"input":record})
                response.raise_for_status()
                result = response.json()
                results.append(result.get('prediction'))
        
        # Create results DataFrame
        result_df = pd.DataFrame(results)
        return JSONResponse(content=result_df.to_dict(orient='records'))
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500) 