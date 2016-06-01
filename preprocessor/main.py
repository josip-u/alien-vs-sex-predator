#import os
#import sys
import pickle

#import nltk
#from nltk.stem.wordnet import WordNetLemmatizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from tfidf_builder import TfidfBuilder
from data_preparation import *
from time import time

print("loading stuff...")

inputs = pickle.load(open("../inputs_dict_new.p", "rb"))
outputs = pickle.load(open("../outputs_dict.p", "rb"))
token_freq_dict = pickle.load(open("token_freq_dict.p", "rb"))
tfidf_builder = pickle.load(open("tfidf_builder.p", "rb"))

print("starting...")

lbound = 5
ubound = 99999
ok_users = filter_by_message_count(inputs, lbound, ubound)
'''
counter = 0
for user in ok_users:
    msgs = len(inputs[user])
    if msgs > 2400:
        counter += 1
print(counter)
'''

documents = to_documents(ok_users)
#tfidf_builder.to_tfidf(documents)

user_duration_dict = {}
for user in ok_users:
    messages = ok_users[user]
    document = to_document(messages)
    start = time()
    tfidf_vector = tfidf_builder.to_tfidf_vector(document)
    end = time()
    user_duration_dict[user] = end - start
pickle.dump(user_duration_dict, open("user_duration_dict.p", "wb"))

#users_vec, input_vec, output_vec = build_classifier_io(ok_users, outputs, tfidf_builder, fit_builder=True)
#print(users_vec.shape, input_vec.shape, output_vec.shape)
#pickle.dump(tfidf_builder, open("tfidf_builder_new.p", "wb"))
