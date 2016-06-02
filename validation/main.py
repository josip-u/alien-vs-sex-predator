import sys
import os
import pickle
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

model = SVC()
parameter_grid = {}
users, inputs, outputs = build_classifier_io(ok_users, outputs, tfidf_builder, fit_builder=False)
cross_validate(inputs, outputs, model, parameter_grid)

print("done")
