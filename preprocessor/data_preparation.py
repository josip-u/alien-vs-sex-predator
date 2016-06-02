import sys
import os
import copy
import numpy as np

import tfidf_builder
sys.path.insert(0, os.path.abspath('../parser'))
import message


def get_filtered_out_words(word_frequency_dict, threshold):
    filtered_out_words = set([])

    for word in word_frequency_dict:
        if word_frequency_dict[word] <= threshold:
            filtered_out_words.add(word)

    return filtered_out_words


def to_document(messages):
    document = ""
    for message in messages:
        document += " " + message._text
    return document.strip()


def to_documents(user_message_dict):
    documents = []

    for user in user_message_dict:
        document = to_document(user_message_dict[user])
        if len(document) > 0:
            documents.append(document)

    return documents


def filter_by_message_count(user_message_dict, lbound, ubound):
    filtered_user_message_dict = {}

    for user in user_message_dict:
        msg_count = len(user_message_dict[user])
        if msg_count > lbound and msg_count < ubound:
            filtered_user_message_dict[user] = user_message_dict[user]

    return filtered_user_message_dict


def filter_by_class_count(user_message_dict, user_class_dict, class_count_dict):
    class_count_dict_copy = copy.deepcopy(class_count_dict)
    filtered_user_message_dict = {}

    for user in user_message_dict:
        user_class = user_class_dict[user]
        if class_count_dict_copy[user_class] > 0:
            filtered_user_message_dict[user] = user_message_dict[user]
            class_count_dict_copy[user_class] -= 1

    return filtered_user_message_dict


def filter_by_id(user_message_dict, user_ids, exclude=False):
    filtered_user_message_dict = {}

    if exclude:
        for user in user_message_dict:
            if user not in user_ids:
                filtered_user_message_dict[user] = user_message_dict[user]
    else:
        for user in user_ids:
            filtered_user_message_dict[user] = user_message_dict[user]

    return filtered_user_message_dict


def filter_by_duration(user_message_dict, user_duration_dict, threshold):
    filtered_user_message_dict = {}

    for user in user_duration_dict:
        if user_duration_dict[user] < threshold:
            filtered_user_message_dict[user] = user_message_dict[user]

    return filtered_user_message_dict


def to_time_vector(messages, time_splits):
    time_vector = [[0 for _ in time_splits]]

    for message in messages:
        time = message._time
        for i in range(len(time_splits)):
            lbound = time_splits[i-1]
            ubound = time_splits[i]
            if time >= lbound and time < ubound:
                time_vector[0][i] += 1
                break
            elif lbound > ubound and (time >= lbound or time < ubound):
                time_vector[0][i] += 1
                break

    max_slot_value = max(time_vector[0])
    for i in range(len(time_vector)):
        time_vector[0][i] = 1 if time_vector[0][i] == max_slot_value else 0

    return np.array(time_vector)


def to_input_vector(messages, tfidf_builder, time_splits):
    document = to_document(messages)
    tfidf_vector = tfidf_builder.to_tfidf_vector(document)
    time_vector = to_time_vector(messages, time_splits)
    return np.concatenate((tfidf_vector, time_vector), axis=1)


def build_classifier_io(user_messages_dict, user_class_dict, tfidf_builder, fit_builder=False, time_splits=[7, 15, 23]):
    if fit_builder:
        documents = to_documents(user_messages_dict)
        tfidf_builder.to_tfidf(documents)

    user_vector = []
    input_vector = []
    output_vector = []

    counter = 1
    total = len(user_messages_dict)
    for user in user_messages_dict:
        print(str(counter) + "/" + str(total))
        counter += 1
        user_vector.append(user)
        messages = user_messages_dict[user]
        input_vector.append(to_input_vector(messages, tfidf_builder, time_splits)[0])
        output_vector.append(user_class_dict[user])

    return user_vector, input_vector, output_vector
