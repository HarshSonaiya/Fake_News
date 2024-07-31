import bentoml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from helper import preprocess_data

if __name__ == "__main__" :

    # Load and preprocess data
    news_dataset = pd.read_csv('data/train.csv')
    news_dataset = preprocess_data(news_dataset)

    X = news_dataset['content']
    Y = news_dataset['label']

    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

    #Create a pipeline with TfidfVectorizer and LogisticRegression
    pipeline = make_pipeline(TfidfVectorizer(),LogisticRegression())
    pipeline.fit(X_train,Y_train)
    
    saved_model = bentoml.sklearn.save_model("fakenewsclassifier",pipeline)

