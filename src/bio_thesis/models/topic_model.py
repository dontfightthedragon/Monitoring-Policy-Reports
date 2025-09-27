
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

def run_bertopic(texts):
    vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=10)
    model = BERTopic(vectorizer_model=vectorizer, calculate_probabilities=True)
    model.fit(texts)
    return model
