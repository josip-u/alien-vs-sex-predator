from nltk import word_tokenize
from time import time


def create_token_freq_dict(user_messages_dict):
    user_duration_dict = {}

    all_tokens = []
    for user in user_messages_dict:
        start = time()
        for message in user_messages_dict[user]:
            all_tokens += word_tokenize(message._text)
        end = time()
        user_duration_dict[user] = end - start

    token_freq_dict = {}
    for token in all_tokens:
        if token in token_freq_dict:
            token_freq_dict[token] += 1
        else:
            token_freq_dict[token] = 0

    return token_freq_dict, user_duration_dict
