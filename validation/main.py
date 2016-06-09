import sys
import os
import pickle
from time import time
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.abspath('../parser'))
sys.path.insert(0, os.path.abspath('../preprocessor'))
from message import *
from tfidf_builder import *
from data_preparation import *
from cross_validation import *


print("loading stuff...")

user_messages_dict = pickle.load(open("../parser/inputs_dict.p", "rb"))
user_class_dict = pickle.load(open("../parser/outputs_dict.p", "rb"))
tfidf_builder = pickle.load(open("../preprocessor/tfidf_builder_new.p", "rb"))
user_duration_dict = pickle.load(open("../preprocessor/user_duration_dict.p", "rb"))

print("starting...")

lbound = 5
ubound = 99999
filtered_user_messages_dict = filter_by_message_count(user_messages_dict, lbound, ubound)
time_threshold = 1000
filtered_user_messages_dict = filter_by_duration(filtered_user_messages_dict, user_duration_dict, time_threshold)

users, inputs, outputs = build_classifier_io(filtered_user_messages_dict, user_class_dict, tfidf_builder)
del user_messages_dict
del user_class_dict
del user_duration_dict
del tfidf_builder
del filtered_user_messages_dict

'''
model_constructor = SVC
# parameter_grid = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-5, 1e-6], 'C': [10000, 100000, 1000000]}]
# parameter_grid = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': [1e-5], 'C': [100000]}]
parameter_grid = [{'kernel': ['linear'], 'gamma': [1e-5], 'C': [10000]}]
'''

model_constructor = LogisticRegression
# parameter_grid = [{'penalty': ['l2'], 'C': [1000, 10000, 100000], 'solver': ['newton-cg', 'lbfgs', 'liblinear']}]
parameter_grid = [{'penalty': ['l2'], 'C': [1000], 'solver': ['newton-cg']}]

'''
model_constructor = KNeighborsClassifier
parameter_grid = [{'n_neighbors': [5, 6, 7], 'weights': ['uniform', 'distance'], }]
'''

# negative_splits = [8, 6, 4]
negative_splits = [4]
scorer = make_scorer(fbeta_score, beta=0.5, average="binary", pos_label=1)
start = time()
cross_validate(inputs, outputs, model_constructor, parameter_grid, scorer=scorer, negative_splits=negative_splits)
end = time()

print("done in", end-start)
