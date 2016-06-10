import os
# import sys
import pickle

# import nltk
# from nltk.stem.wordnet import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from tfidf_builder import TfidfBuilder
from data_preparation import *
from time import time
from premain import *


print("loading stuff...")

user_messages_dict = pickle.load(open("../parser/inputs_dict.p", "rb"))
user_class_dict = pickle.load(open("../parser/outputs_dict.p", "rb"))

token_freq_dict_path = "token_freq_dict.p"
user_duration_dict_path = "user_duration_dict.p"
if os.path.exists(token_freq_dict_path):
    token_freq_dict = pickle.load(open(token_freq_dict_path, "rb"))
    user_duration_dict = pickle.load(open(user_duration_dict_path, "rb"))
else:
    print("gathering tokenization data...")
    token_freq_dict, user_duration_dict = create_tokenization_report(user_messages_dict)
    pickle.dump(token_freq_dict, open(token_freq_dict_path, "wb"))
    pickle.dump(user_duration_dict, open(user_duration_dict, "wb"))

tfidf_builder_path = "tfidf_builder.p"
if os.path.exists(tfidf_builder_path):
    tfidf_builder = pickle.load(open(tfidf_builder_path, "rb"))
    fit_builder = False
else:
    print("creating tfidf_builder...")
    word_count_threshold = 5
    filtered_out_words = get_filtered_out_words(word_frequency_dict, word_count_threshold)
    tfidf_builder = TfidfBuilder(filtered_out_words)
    fit_builder = True


print("starting...")

lbound = 9
ubound = 99999
ok_users = filter_by_message_count(user_messages_dict, lbound, ubound)
time_threshold = 1000
ok_users = filter_by_duration(ok_users, user_duration_dict, time_threshold)

start = time()
if fit_builder:
    documents = to_documents(ok_users)
    tfidf_builder.to_tfidf(documents)
    end = time()
print("Time:", end-start)
pickle.dump(tfidf_builder, open(tfidf_builder_path, "wb"))

# users_vec, input_vec, output_vec = build_classifier_io(ok_users, user_class_dict, tfidf_builder, fit_builder=fit_builder)
# print(len(users_vec), len(input_vec), input_vec[0].shape, len(output_vec))
# print(users_vec.shape, input_vec.shape, output_vec.shape)

