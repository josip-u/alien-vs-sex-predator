import sys
import os
import pickle
from time import time
from sklearn.metrics import fbeta_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.abspath('../parser'))
sys.path.insert(0, os.path.abspath('../preprocessor'))
from message import *
from tfidf_builder import *
from data_preparation import *


def splits_dicts(user_messages_dict, user_class_dict, part_size=10000):
    user_messages_parts = []
    user_class_parts = []
    current_user_messages_dict = {}
    current_user_class_dict = {}

    for user in user_messages_dict:
        if len(current_user_messages_dict) >= part_size:
            user_messages_parts.append(current_user_messages_dict)
            user_class_parts.append(current_user_class_dict)
            current_user_messages_dict = {}
            current_user_class_dict = {}

        current_user_messages_dict[user] = user_messages_dict[user]
        current_user_class_dict[user] = user_class_dict[user]

    if len(current_user_class_dict) > 0:
        user_messages_parts.append(current_user_messages_dict)
        user_class_parts.append(current_user_class_dict)

    return user_messages_parts, user_class_parts


print("loading stuff...")

user_messages_dict = pickle.load(open("../parser/test_inputs_dict.p", "rb"))
user_class_dict = pickle.load(open("../parser/test_outputs_dict.p", "rb"))
tfidf_builder = pickle.load(open("../preprocessor/tfidf_builder_new.p", "rb"))
# classifier = pickle.load(open("../validation/best_classifier.p", "rb"))
classifier = pickle.load(open("../validation/best_classifier_new.p", "rb"))

print("starting...")

print("splitting dicts...")
user_messages_parts, user_class_parts = splits_dicts(user_messages_dict, user_class_dict, part_size=1000)
del user_messages_dict
del user_class_dict

all_users = None
all_outputs = None
all_predicted = None

counter = 0
total = len(user_messages_parts)
for user_messages_dict, user_class_dict in zip(user_messages_parts, user_class_parts):
    users, inputs, outputs = build_classifier_io(user_messages_dict, user_class_dict, tfidf_builder)
    print("predicting (" + str(counter+1) + "/" + str(total) + ")...")
    predicted = classifier.predict(inputs)

    all_users = users if all_users is None else np.concatenate((all_users, users))
    all_outputs = outputs if all_outputs is None else np.concatenate((all_outputs, outputs))
    all_predicted = predicted if all_predicted is None else np.concatenate((all_predicted, predicted))

    counter += 1
print(all_users.shape, all_outputs.shape, all_predicted.shape)

print("calculating score...")
beta = 0.5
score = fbeta_score(all_outputs, all_predicted, beta=beta)
print("F" + str(beta) + " score is", score)
print("Report:\n", classification_report(all_outputs, all_predicted))
print("Confusion matrix:\n", confusion_matrix(all_outputs, all_predicted, labels=[1, 0]))


best_test_score_path = "best_test_score.p"
postively_predicted_users_path = "postively_predicted_users.p"

if not os.path.exists(best_test_score_path) or score > pickle.load(open(best_test_score_path)):
    print("Saving results...")
    postively_predicted_users = set([])
    for user, predicted in zip(all_users, all_predicted):
        if predicted == 1:
            postively_predicted_users.add(user)

    pickle.dump(postively_predicted_users, open(postively_predicted_users_path, "wb"))
    pickle.dump(postively_predicted_users, open(best_test_score_path, "wb"))
