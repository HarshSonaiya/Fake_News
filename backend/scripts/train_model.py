import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import bentoml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def ensure_stopwords_downloaded():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

ensure_stopwords_downloaded()

porter_stem = PorterStemmer()

def preprocess_data(df):
    df.replace('NaN', np.nan, inplace=True)
    df = df.fillna('')
    df['content'] = df['author'] + ' ' + df['title'] + ' ' + df['text']
    df['content'] = df['content'].apply(stemming)
    return df

digit_re = re.compile(r'\d')
non_word_re = re.compile(r'[^\w\s]')
stop_words = set(stopwords.words('english'))

def stemming(content):
    content = digit_re.sub(' ', content)
    content = non_word_re.sub(' ', content).lower()
    stemmed_content = [porter_stem.stem(word) for word in content.split() if word not in stop_words]
    return ' '.join(stemmed_content)

if __name__ == "__main__" :

    # Load and preprocess data
    news_dataset = pd.read_csv('backend/data/train.csv')
    news_dataset = preprocess_data(news_dataset)

    # Convert text to numerical data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(news_dataset['content'])
    Y = news_dataset['label']

    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)


    saved_model = bentoml.sklearn.save_model("FakeNewsClassifier",model)

    # Evaluate the model
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))

