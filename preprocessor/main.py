#import os
#import sys
import pickle

#import nltk
#from nltk.stem.wordnet import WordNetLemmatizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from tfidf_builder import TfidfBuilder
from data_preparation import *
from time import time

print("loading stuff...")

inputs = pickle.load(open("../inputs_dict_new.p", "rb"))
outputs = pickle.load(open("../outputs_dict.p", "rb"))
token_freq_dict = pickle.load(open("token_freq_dict.p", "rb"))
tfidf_builder = pickle.load(open("tfidf_builder_new.p", "rb"))
user_duration_dict = pickle.load(open("user_duration_dict.p", "rb"))

print("starting...")

lbound = 5
ubound = 99999
ok_users = filter_by_message_count(inputs, lbound, ubound)
time_threshold = 1000
ok_users = filter_by_duration(ok_users, user_duration_dict, time_threshold)

#documents = to_documents(ok_users)
#tfidf_builder.to_tfidf(documents)
'''
bad_users = set([])
for user in user_duration_dict:
    user_time = user_duration_dict[user]
    if user_time > 1000:
        bad_users.add(user)
ok_users = filter_by_id(ok_users, bad_users, exclude=True)
'''

start = time()
users_vec, input_vec, output_vec = build_classifier_io(ok_users, outputs, tfidf_builder, fit_builder=False)
end = time()
print(end-start)
print(len(users_vec), len(input_vec), input_vec[0].shape, len(output_vec))
#pickle.dump(tfidf_builder, open("tfidf_builder_new.p", "wb"))

