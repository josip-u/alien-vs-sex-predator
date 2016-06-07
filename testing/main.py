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


print("loading stuff...")

user_messages_dict = pickle.load(open("../parser/test_inputs_dict.p", "rb"))
user_class_dict = pickle.load(open("../parser/test_outputs_dict.p", "rb"))
tfidf_builder = pickle.load(open("../preprocessor/tfidf_builder_new.p", "rb"))
classifier = pickle.load(open("../validation/best_classifier.p", "rb"))

print("starting...")

start = time()
users, inputs, outputs = build_classifier_io(user_messages_dict, user_class_dict, tfidf_builder)
end = time()
print("done in", end-start)

print("calculating score...")
scorer = make_scorer(fbeta_score, beta=0.5, average="binary", pos_label=1)
score = scorer(classifier, inputs, outputs)
print("Fbeta score is", score)
