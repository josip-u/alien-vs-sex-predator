import sys
import os
import pickle
import numpy as np
from sklearn.metrics import fbeta_score

sys.path.insert(0, os.path.abspath('../parser'))
sys.path.insert(0, os.path.abspath('../preprocessor'))
from message import *
from data_preparation import *

print("loading stuff...")

user_messages_dict = pickle.load(open("../parser/test_inputs_dict.p", "rb"))
positively_predicted_users = pickle.load(open("positively_predicted_users.p", "rb"))
lines = pickle.load(open("../parser/lines.p", "rb"))
lines = set(lines)
classifier = pickle.load(open("../validation/best_classifier.p", "rb"))
# classifier = pickle.load(open("../validation/best_classifier_new.p", "rb"))
tfidf_builder = pickle.load(open("../preprocessor/tfidf_builder_new.p", "rb"))

print("starting...")

all_messages = []
positive_messages = set([])
for user in user_messages_dict:
    if user in positively_predicted_users:
        for message in user_messages_dict[user]:
            positive_messages.add((message._conversation_id, message._message_line))
            all_messages.append(message)

tp = lines & positive_messages
fp = positive_messages - lines
fn = lines - positive_messages
print("TP:", len(tp), "FP:", len(fp), "FN:", len(fn))


print("building input/output vector...")

messages_output = []
input_vectors = []
counter = 0
total = len(all_messages)
for message in all_messages:
    print(str(counter+1) + "/" + str(total))
    counter += 1

    msg_id = (message._conversation_id, message._message_line)
    output = 1 if msg_id in lines else 0
    messages_output.append(output)

    tfidf_vector = tfidf_builder.to_tfidf_vector(message._text)
    time_vector = to_time_vector([message])
    input_vector = np.concatenate((tfidf_vector, time_vector), axis=1)
    input_vectors.append(input_vector)


beta = 3
score = fbeta_score(messages_output, classifier.predict(input_vectors), beta=beta)
print("F" + str(beta) + " score is", score)
