import pickle
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


def data_split(inputs, outputs, split_count, run_all_splits):
    input_splits = [[] for _ in range(split_count)]
    output_splits = [[] for _ in range(split_count)]

    index = 0
    for input_, output_ in zip(inputs, outputs):
        if output_ == 1:
            for input_list, output_list in zip(input_splits, output_splits):
                input_list.append(input_)
                output_list.append(output_)
        else:
            input_splits[index].append(input_)
            output_splits[index].append(output_)
            index = (index + 1) % split_count

    if run_all_splits:
        return input_splits, output_splits
    else:
        return input_splits[:1], output_splits[:1]


def get_best_classifier(classifiers, inputs, outputs, scorer):
    best_score = 0
    best_classifier = None
    for classifier in classifiers:
        score = scorer(classifier, inputs, outputs)
        if score > best_score or best_classifier is None:
            best_score = score
            best_classifier = classifier
        print("Fbeta score:", score)
    print()

    current_best_score = pickle.load(open("best_score.p", "rb"))
    if best_score > current_best_score:
        print("pickling new champion (" + str(best_score) + " > " + str(current_best_score) + ")...")
        pickle.dump(best_score, open("best_score.p", "wb"))
        pickle.dump(best_classifier, open("best_classifier.p", "wb"))

    return best_classifier


def cross_validate(inputs, outputs, model_constructor, parameters, scorer, train_perc=0.7,
                   negative_splits=[16, 8, 4, 2, 1], folds=5, n_jobs=1, run_all_splits=False):
    test_size = 1.0 - train_perc
    train_in, test_in, train_out, test_out = train_test_split(inputs, outputs, test_size=test_size, stratify=outputs,
                                                              random_state=1337)

    classifiers = []
    for splits_count in negative_splits:
        print("Computing for part size: 1/" + str(splits_count) + "...")

        splits_in, splits_out = data_split(train_in, train_out, splits_count, run_all_splits)
        for split_in, split_out in zip(splits_in, splits_out):
            clf = GridSearchCV(model_constructor(), parameters, cv=folds, scoring=scorer, n_jobs=n_jobs, refit=True)
            clf.fit(split_in, split_out)
            classifiers.append(clf)

            print("\tBest score:", clf.best_score_)
            print("\tBest params:", clf.best_params_)

        # print("break2")
        # break

    best_classifier = get_best_classifier(classifiers, test_in, test_out, scorer)
    prediction_out = best_classifier.predict(test_in)
    print(classification_report(test_out, prediction_out))
    print(best_classifier.best_params_)
    return best_classifier
