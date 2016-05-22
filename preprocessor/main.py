import os
import sys
import pickle

#import nltk
from nltk.stem.wordnet import WordNetLemmatizer
#from sklearn.feature_extraction.text import TfidfVectorizer

from tfidf_builder import TfidfBuilder
from data_filtering import *


'''
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
'''


inputs = pickle.load(open("../inputs_dict_new.p", "rb"))
outputs = pickle.load(open("../outputs_dict.p", "rb"))
token_freq_dict = pickle.load(open("token_freq_dict.p", "rb"))


user_count = len(inputs)
for msg_treshold in range(1, 11):
    treshold_count = 0
    lost_count = 0
    for user in inputs:
        count = len(inputs[user])
        if count <= msg_treshold:
            treshold_count += 1
            if outputs[user] == 1:
                lost_count += 1

    print(msg_treshold, treshold_count, user_count, treshold_count * 100 / user_count, lost_count)
print()

word_count = len(token_freq_dict)
for msg_treshold in range(1, 6):
    treshold_count = 0
    for token in token_freq_dict:
        count = token_freq_dict[token]
        if count <= msg_treshold:
            treshold_count += count

    print(msg_treshold, treshold_count, word_count, treshold_count * 100 / word_count)
print()

non_word_count = 0
for word in token_freq_dict:
    if not word.isalnum():
        non_word_count += 1
print(non_word_count, word_count, non_word_count * 100 / word_count)
non_word_count = 0
non_filtered_count = 0
for word in token_freq_dict:
    if token_freq_dict[word] <= 5:
        continue
    non_filtered_count += 1
    if not word.isalnum():
        non_word_count += 1
print(non_word_count, non_filtered_count, non_word_count * 100 / non_filtered_count)
print()


threshold = 5
filtered_out_words = get_filtered_out_words(token_freq_dict, threshold)
tfidf_builder = TfidfBuilder(filtered_out_words)

ok_users = filter_by_message_count(inputs, threshold)
documents = to_documents(ok_users)
#tfidf_builder.to_tfidf(documents)
#vec = tfidf_builder.to_tfidf_vector(documents[0])
#print(type(vec), len(vec))

class_count_dict = {}
for user in ok_users:
    user_class = outputs[user]
    if user_class in class_count_dict:
        class_count_dict[user_class] += 1
    else:
        class_count_dict[user_class] = 0
print(class_count_dict)
print()

class_count_dict[0] = 120
class_count_dict[1] = 120
while class_count_dict[0] <= 20954:
    ok_users2 = filter_by_class_count(ok_users, outputs, class_count_dict)
    documents = to_documents(ok_users2)
    tfidf_builder.to_tfidf(documents)
    vec = tfidf_builder.to_tfidf_vector(documents[0])
    print(class_count_dict[0], vec.shape)
    class_count_dict[0] *= 2
print()


