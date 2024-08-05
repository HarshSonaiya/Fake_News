import bentoml
import pandas as pd
from bentoml.io import JSON, File
from bentoml.exceptions import BentoMLException
from pydantic import BaseModel
from helper import preprocess_data
import io

class Input(BaseModel):
    title: str
    author: str
    text: str

@bentoml.service(
    resources={"cpu": "2"}
)
class Prediction:
    '''
    A simple news classifier service using TF-IDF Vectorizer and Logistic Regression
    '''

    # Load the saved model as a runner
    model_runner = bentoml.sklearn.get("fakenewsclassifier:latest")

    def __init__(self):
        '''
        Importing the service by loading the model from the model store.
        '''
        import joblib

        self.model = joblib.load(self.model_runner.path_of("saved_model.pkl"))

    @bentoml.api()
    async def predict(
        self,
        input: dict,
    ):
        '''
        Defining an API endpoint to serve incoming FAST API requests 
        '''
        try:
            '''
            Convert the input to dictionary and then convert dictionary to DataFrame
            to utilize keys. Then preprocess the data and apply stemming in order to 
            vectorize the input and make predictions. 
            '''
            df = pd.DataFrame([input])
            processed_text = preprocess_data(df)

            X_test = processed_text['content']
            prediction = self.model.predict(X_test)

            prediction = ['real' if pred == 1 else 'fake' for pred in prediction]
            return {"prediction": prediction}
    
        except BentoMLException as e:
            return {"error": str(e)}

    @bentoml.api()
    async def predict_file(
            self,
            inputs: dict
        ):
        '''
            Defining an API endpoint to serve incoming CSV file requests
        '''
        try: 
            df = pd.DataFrame([inputs])
            processed_text = preprocess_data(df)
            X_test = processed_text['content']
            prediction = self.model.predict(X_test)

            prediction = ['real' if pred == 1 else 'fake' for pred in prediction]
            results = [{"title": row['title'], "prediction": pred} for row, pred in zip(df.to_dict(orient='records'), prediction)]

            return results      
        except Exception as e:
            return {"error": str(e)}