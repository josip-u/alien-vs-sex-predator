import sys
import os
import copy
sys.path.insert(0, os.path.abspath('../parser'))
import message


def get_filtered_out_words(word_frequency_dict, threshold):
    filtered_out_words = set([])

    for word in word_frequency_dict:
        if word_frequency_dict[word] <= threshold:
            filtered_out_words.add(word)

    return filtered_out_words


def to_documents(user_message_dict):
    documents = []

    for user in user_message_dict:
        document = ""
        for message in user_message_dict[user]:
            document += " " + message._text
        document = document.strip()
        if len(document) > 0:
            documents.append(document)

    return documents


def filter_by_message_count(user_message_dict, threshold):
    filtered_user_message_dict = {}

    for user in user_message_dict:
        if len(user_message_dict[user]) > threshold:
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
