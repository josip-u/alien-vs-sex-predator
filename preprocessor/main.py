#import os
#import sys
import pickle

#import nltk
#from nltk.stem.wordnet import WordNetLemmatizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from tfidf_builder import TfidfBuilder
from data_preparation import *

print("loading stuff...")

inputs = pickle.load(open("../inputs_dict_new.p", "rb"))
outputs = pickle.load(open("../outputs_dict.p", "rb"))
token_freq_dict = pickle.load(open("token_freq_dict.p", "rb"))
tfidf_builder = pickle.load(open("tfidf_builder.p", "rb"))

print("starting...")

threshold = 5
ok_users = filter_by_message_count(inputs, threshold)

counter = 0
for user in ok_users:
    msgs = len(inputs[user])
    if msgs > 2400:
        counter += 1
print(counter)

'''
users_vec, input_vec, output_vec = build_classifier_io(ok_users, outputs, tfidf_builder)
print(users_vec.shape, input_vec.shape, output_vec.shape)
pickle.dump((users_vec, input_vec, output_vec), open("users_inputs_outputs.p", "wb"))
'''
