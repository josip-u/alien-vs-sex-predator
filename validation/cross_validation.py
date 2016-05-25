def cross_validate(user_messages_dict, user_class_dict, train_perc=0.7, negative_shares=[16, 8, 4, 2, 1]):
    positive_users = set([])
    negative_users = set([])
    for user in user_class_dict:
        if user_class_dict[user] == 0:
            negative_users.add(user)
        else:
            positive_users.add(user)

    train_users = set([])
    test_users = set([])
    positive_train_count = int(0.7 * len(positive_users))
    negative_train_count = int(0.7 * len(negative_users))
    for user in user_class_dict:
        if user_class_dict[user] == 0:
            if negative_train_count > 0:
                train_users.add(user)
                negative_train_count -= 1
            else:
                test_users.add(user)
        else:
            if positive_train_count > 0:
                train_users.add(user)
                positive_train_count -= 1
            else:
                test_users.add(user)

    for negative_share in negative_shares:
        pass
