import sys
import os
import pickle
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.abspath('../parser'))
sys.path.insert(0, os.path.abspath('../preprocessor'))
from message import *
from data_preparation import *

print("loading stuff...")

user_messages_dict = pickle.load(open("../parser/test_inputs_dict.p", "rb"))
positively_predicted_users = pickle.load(open("positively_predicted_users.p", "rb"))
positive_messages = pickle.load(open("../parser/lines.p", "rb"))
positive_messages = set(positive_messages)
classifier = pickle.load(open("../validation/best_classifier.p", "rb"))
# classifier = pickle.load(open("../validation/best_classifier_new.p", "rb"))
tfidf_builder = pickle.load(open("../preprocessor/tfidf_builder_new.p", "rb"))

print("starting...")

all_messages = []
positively_predicted_messages = set([])
for user in user_messages_dict:
    if user in positively_predicted_users:
        for message in user_messages_dict[user]:
            positively_predicted_messages.add((message._conversation_id, message._message_line))
            all_messages.append(message)

tp = positive_messages & positively_predicted_messages
fp = positively_predicted_messages - positive_messages
fn = positive_messages - positively_predicted_messages
print("TP:", len(tp), "FP:", len(fp), "FN:", len(fn))


print("building input/output vectors...")

messages_output = []
for message in all_messages:
    msg_id = (message._conversation_id, message._message_line)
    output = 1 if msg_id in positive_messages else 0
    print(output)
    messages_output.append(output)

messages_predicted_outputs = None

step = 1000
total = len(all_messages)

for index in range(0, total, step):
    print(str(index) + "/" + str(total))
    for message in all_messages[index : index+step]:
        input_vectors = []
        tfidf_vector = tfidf_builder.to_tfidf_vector(message._text)
        time_vector = to_time_vector([message])
        input_vector = np.concatenate((tfidf_vector, time_vector), axis=1)
        input_vectors.append(input_vector[0])
        output = classifier.predict(input_vectors)
        messages_predicted_outputs = output if messages_predicted_outputs is None else np.concatenate((messages_predicted_outputs, output))

beta = 3
score = fbeta_score(messages_output, messages_predicted_outputs, beta=beta)
print("F" + str(beta) + " score is", score)
print("Report:\n", classification_report(messages_output, messages_predicted_outputs))
print("Confusion matrix:\n", confusion_matrix(messages_output, messages_predicted_outputs, labels=[1, 0]))
