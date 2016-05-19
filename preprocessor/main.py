import nltk
import os
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def get_tokens(text):
    text = text.lower()
    all_tokens = nltk.word_tokenize(text)
    word_tokens = [word for word in all_tokens]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in word_tokens]
    return lemmatized_tokens


def read_directory_files(path):
    file_text_dict = {}

    for directory_path, directories, files in os.walk(path):
        for file in files:
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(directory_path, file)
            file = open(file_path, "r")
            file_text_dict[file] = file.read()

    return file_text_dict


def to_tfidf(documents):
    tfidf = TfidfVectorizer(tokenizer=get_tokens)
    tfidf.fit(documents)
    return tfidf


def to_tfidf_vector(tfidf, document):
    return tfidf.transform([document]).toarray()


file_text_dict = read_directory_files(".")
tfidf = to_tfidf(file_text_dict.values())

inputs = pickle.load(open("../inputs_dict.p", "rb"))
outputs = pickle.load(open("../outputs_dict.p", "rb"))
token_freq_dict = pickle.load(open("../token_freq_dict.p", "rb"))


user_count = len(inputs)
for msg_treshold in range(1, 6):
    treshold_count = 0
    for user in inputs:
        count = len(inputs[user])
        if count <= msg_treshold:
            treshold_count += 1

    print(msg_treshold, treshold_count, user_count, treshold_count * 100 / user_count)
print()

word_count = len(token_freq_dict)
for msg_treshold in range(1, 6):
    treshold_count = 0
    for token in token_freq_dict:
        count = token_freq_dict[token]
        if count <= msg_treshold:
            treshold_count += count

    print(msg_treshold, treshold_count, word_count, treshold_count * 100 / word_count)

