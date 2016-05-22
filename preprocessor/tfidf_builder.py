import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfBuilder:

    def __init__(self, filtered_out_words=[]):
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf = TfidfVectorizer(tokenizer=self.get_tokens)
        self.filtered_out_words = filtered_out_words

    def filter(self, word):
        result = True
        if word in self.filtered_out_words:
            result = False
        return result

    def get_tokens(self, text):
        all_tokens = nltk.word_tokenize(text)
        filtered_tokens = [word for word in all_tokens if self.filter(word)]
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
        return lemmatized_tokens

    def to_tfidf(self, documents):
        self.tfidf.fit(documents)
        return self.tfidf

    def to_tfidf_vector(self, document):
        return self.tfidf.transform([document]).toarray()
