import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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