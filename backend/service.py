import bentoml
import pandas as pd
from bentoml.io import JSON
from bentoml.exceptions import BentoMLException
from pydantic import BaseModel

class Input(BaseModel):
    text: str 

# Load the saved model as a runner
model_runner = bentoml.sklearn.get("fakenewsclassifier:latest").to_runner()

service = bentoml.Service("classifier",runners=[model_runner])  

'''
Problem 1:
Defining the service like this results into error that resources is not an attribute
@bentoml.Service(name="News_Classifier", 
                        runners=[model_runner],
                        resources={"cpu": "200m", "memory": "512Mi"}
                    )
'''

'''
Problem 2:
To specify resources we can use the following :

model_runner = bentoml.sklearn.get("fakenewsclassifier:latest").to_runner()
@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
    runners=[model_runner],
    name="News-Classifier"
)
@bentoml.api
def predict(input: Input):
           #prediction logic

The issue here is that input and output format for api cannot be sprecified.
'''

'''
Problem 3:
Even after mentioning the train_model.py as a package and to prevent unnecessary 
imports keeping the statements other than present inside the function inside condition
if __name__ == "__main__" they are executed 
'''

'''
Problem 4:
The other issue is if the name specified in 
service = bentoml.Service("classifier",runners=[model_runner])  
is in upper case multiple calls (here 16) like the one below are made instead of 1:
2024-07-30T18:12:50+0530 [WARNING] [api_server:16] Converting News-Classifier to 
lowercase: news-classifier.
'''

'''
Problem 5:
This all combined causes the vscode and windows to crash and after removing these 
and executing bentoml serve service:service the execution starts and does not cause 
VS Code to crash but also does not load the url http://localhost:3000
'''

'''
Probelm 6:
And by attempting to run using bentoml build cause the errors from above.
'''

# Define the predict API endpoint
@service.api(input=JSON(),output=JSON())
def predict(input: str):
    try:
        df = pd.DataFrame([input])
        # processed_text = preprocess_data(df)

        # X_test = model_runner.artifacts["tfidf_vectorizer"].transform(processed_text['content'])
        # prediction = model_runner.artifacts["model"].predict(X_test)

        # prediction = ['real' if pred == 1 else 'fake' for pred in prediction]
        return {"prediction": 1}
    
    except BentoMLException as e:
        return {"error": str(e)}
