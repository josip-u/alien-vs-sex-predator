import sys
import os
import pickle
from time import time
from sklearn.svm import SVC

sys.path.insert(0, os.path.abspath('../parser'))
sys.path.insert(0, os.path.abspath('../preprocessor'))
from message import *
from tfidf_builder import *
from data_preparation import *
from cross_validation import *


print("loading stuff...")

user_messages_dict = pickle.load(open("../inputs_dict_new.p", "rb"))
user_class_dict = pickle.load(open("../outputs_dict.p", "rb"))
tfidf_builder = pickle.load(open("../preprocessor/tfidf_builder_new.p", "rb"))
user_duration_dict = pickle.load(open("../preprocessor/user_duration_dict.p", "rb"))

print("starting...")

lbound = 5
ubound = 99999
filtered_user_messages_dict = filter_by_message_count(user_messages_dict, lbound, ubound)
time_threshold = 1000
filtered_user_messages_dict = filter_by_duration(filtered_user_messages_dict, user_duration_dict, time_threshold)

users, inputs, outputs = build_classifier_io(filtered_user_messages_dict, user_class_dict, tfidf_builder, fit_builder=False)
del user_messages_dict
del user_class_dict
del user_duration_dict
del tfidf_builder
del filtered_user_messages_dict

model_constructor = SVC
parameter_grid = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
negative_splits = [64, 32, 16, 8, 4]
start = time()
cross_validate(inputs, outputs, model_constructor, parameter_grid, negative_splits=negative_splits)
end = time()

print("done in", end-start)
