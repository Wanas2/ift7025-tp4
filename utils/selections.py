import random
import numpy as np

def train_test_split(X, y, train_ratio, random_state=None):
    data = np.concatenate((np.copy(X), np.copy(y)[np.newaxis, :].T), axis=1)

    if isinstance(random_state, int):
        np.random.seed(random_state)

    np.random.shuffle(data)

    size = int(train_ratio * data.shape[0] / 100)

    train, train_labels = np.copy(data[:size+1, :-1]), np.copy(data[:size+1, -1])
    test, test_labels = np.copy(data[size+1:, :-1]), np.copy(data[size+1:, -1])

    return train, train_labels, test, test_labels

def KFold(X, n_splits=10, shuffle=False, random_state=None):
    X_split = list()
    X_copy = list(X)
    
    fold_size = int(len(X) / n_splits)
    
    for _ in range(n_splits):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(X_copy))
            X_copy.pop(index)
            fold.append(index)
        X_split.append(fold)

    result = list()
    for n in range(n_splits):
        X_copy = list(X_split)
        test_index = X_copy.pop(n)
        train_index = X_copy
        result.append((train_index, test_index))
    
    return result

def cross_validation_scores(estimator, X, y, n_folds):
    scores = list()
    for train_index, test_index in KFold(X, n_folds):
        X_train = list()
        y_train = list()
        for index in train_index:
            X_train = X_train + list(X[index])
            y_train = y_train + list(y[index])

        X_test = X[test_index]
        y_test = y[test_index]

        estimator.train(X_train, y_train)
        
        evals = estimator.evaluate(X_test, y_test)
        del(evals["con_matrix"])
        scores.append(evals)

    return scores