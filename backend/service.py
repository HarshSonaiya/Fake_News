import bentoml
import pandas as pd
from bentoml.io import JSON
from bentoml.exceptions import BentoMLException
from pydantic import BaseModel
from helper import preprocess_data

class Input(BaseModel):
    title: str
    author: str
    text: str 

@bentoml.service(
    resources={"cpu": "2"}
)

class Prediction:
    '''
    A simple news classifier service using TF-IDF Vectorizer and Logistice Regression
    '''

    # Load the saved model as a runner
    model_runner = bentoml.sklearn.get("fakenewsclassifier:latest")

    def __init__(self):
        '''
        importing the service by loading the model from the model strore.
        '''
        import joblib

        self.model = joblib.load(self.model_runner.path_of("saved_model.pkl"))

    @bentoml.api
    async def predict(
        self,
        input: Input,
    ):
        '''
        Defining an API endpoint to serve incoming FAST API requests 
        '''
        try:
            '''
            Convert the input to dictionary and then convert dictionary to Dataframe
            to utilize keys.Then preprocess the data and apply stemming in order to 
            vectorize the input and make preditions. 
            '''
            df = pd.DataFrame([input.model_dump()])
            processed_text = preprocess_data(df)

            X_test = processed_text['content']
            prediction = self.model.predict(X_test)

            prediction = ['real' if pred == 1 else 'fake' for pred in prediction]
            return {"prediction": prediction}
    
        except BentoMLException as e:
            return {"error": str(e)}

