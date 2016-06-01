import scipy as sp
from sklearn.metrics import log_loss


def cross_validate(user_messages_dict, user_class_dict, train_perc=0.7, negative_shares=[16, 8, 4, 2, 1], folds=5):
    positive_users = set([])
    negative_users = set([])
    for user in user_class_dict:
        if user_class_dict[user] == 0:
            negative_users.add(user)
        else:
            positive_users.add(user)

    train_users = set([])
    test_users = set([])
    positive_train_count = int(train_perc * len(positive_users))
    negative_train_count = int(train_perc * len(negative_users))
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
        share_count = int(negative_train_count * (1/negative_share))
        pass


def grid_search(model, X_train, X_validate, y_train, y_validate, c1_c2, g1_g2, error_surface=False):
    c1, c2 = c1_c2;
    g1, g2 = g1_g2
    C = sp.array([2 ** ci for ci in range(c1, c2 + 1)])
    G = sp.array([2 ** gi for gi in range(g1, g2 + 1)])
    errors_train = sp.zeros((len(C), len(G)))
    errors_validate = sp.zeros((len(C), len(G)))

    best_ij = (0, 0)
    for i, c in enumerate(C):
        for j, g in enumerate(G):
            model.C = c
            model.gamma = g
            model.fit(X_train, y_train)
            errors_train[i, j] = log_loss(y_train, model.predict(X_train))
            errors_validate[i, j] = log_loss(y_validate, model.predict(X_validate))
            if errors_validate[i, j] < errors_validate[best_ij]:
                best_ij = (i, j)

    best_c = C[best_ij[0]]
    best_g = G[best_ij[1]]
    if error_surface:
        return best_c, best_g, errors_train, errors_validate
    else:
        return best_c, best_g
