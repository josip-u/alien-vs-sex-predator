# import os
# import sys
import pickle

# import nltk
# from nltk.stem.wordnet import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from tfidf_builder import TfidfBuilder
from data_preparation import *
from time import time

print("loading stuff...")

inputs = pickle.load(open("../parser/inputs_dict_new.p", "rb"))
outputs = pickle.load(open("../parser/outputs_dict.p", "rb"))
token_freq_dict = pickle.load(open("token_freq_dict.p", "rb"))
tfidf_builder = pickle.load(open("tfidf_builder_new.p", "rb"))
user_duration_dict = pickle.load(open("user_duration_dict.p", "rb"))

print("starting...")

lbound = 9
ubound = 99999
ok_users = filter_by_message_count(inputs, lbound, ubound)
time_threshold = 1000
ok_users = filter_by_duration(ok_users, user_duration_dict, time_threshold)

'''
word_count_threshold = 5
filtered_out_words = get_filtered_out_words(word_frequency_dict, word_count_threshold)
tfidf_builder = TfidfBuilder(filtered_out_words)
'''

start = time()
users_vec, input_vec, output_vec = build_classifier_io(ok_users, outputs, tfidf_builder, fit_builder=False)
end = time()
print(end-start)
# print(len(users_vec), len(input_vec), input_vec[0].shape, len(output_vec))
print(users_vec.shape, input_vec.shape, output_vec.shape)
# pickle.dump(tfidf_builder, open("tfidf_builder_new.p", "wb"))

