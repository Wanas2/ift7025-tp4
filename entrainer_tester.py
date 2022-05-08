import numpy as np

import load_datasets
import DecisionTree
import NeuralNet

#importer d'autres fichiers et classes si vous en avez développés
import utils.graphics as graphics
import utils.selections as selections
import utils.metrics as metrics

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

import time

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester
"""

# Charger/lire les datasets
iris = load_datasets.load_iris_dataset(92)
wine = load_datasets.load_wine_dataset(70)
abalone = load_datasets.load_abalone_dataset(70)

datasets = [iris, wine, abalone]

print("DecisionTree")

# graphics.plot_learning_curve(iris, DecisionTree.DecisionTreeClassifier())

print("============ Custom: ")
for index, data in enumerate(datasets):
    print("Dataset", {0:"iris", 1:"wine", 2:"abalone"}[index])
    dT = DecisionTree.DecisionTreeClassifier()

    X_train, y_train, X_test, y_test = selections.train_test_split(data[0].astype(np.float64), data[1], 70, random_state=66)

    t1 = time.time()
    dT.train(X_train.astype(np.float64), y_train)
    print("\nAvant élagage")
    print("n_leaves = {}, depth = {}".format(dT.count_leaves(dT.tree), dT.get_depth(dT.tree)))
    t2 = time.time()

    if index == 0:
        graphics.plot_tree(dT.tree)

    t3 = time.time()
    print("Evaluation:")
    print("Train")
    print(dT.evaluate(X_train.astype(np.float64), y_train))
    print("Test")
    print(dT.evaluate(data[2].astype(np.float64), data[3]))
    t4 = time.time()

    dT.tree = dT.prune(dT.tree, X_test.astype(np.float64), y_test)
    print("\nAprès élagage")
    print("n_leaves = {}, depth = {}".format(dT.count_leaves(dT.tree), dT.get_depth(dT.tree)))
    
    print("Evaluation:")
    print("Train")
    print(dT.evaluate(X_train.astype(np.float64), y_train))
    print("Test")
    print(dT.evaluate(data[2].astype(np.float64), data[3]))

    print("Temps d'entrainement: {0}\nTemps d'évaluation: {1}\n".format((t2-t1), (t4-t3)))

print("\n")

print("============ Sklearn: ")
for index, data in enumerate(datasets):
    print("Dataset", {0:"iris", 1:"wine", 2:"abalone"}[index])
    dT = DecisionTreeClassifier(criterion="entropy")

    dT.fit(data[0].astype(np.float64), data[1])
    if index == 0:
        plot_tree(dT)
        plt.show()

    scores = dict()

    y_pred = dT.predict(data[2].astype(np.float64))
    y_true = data[3]

    print(metrics.all_metrics(y_pred, y_true, np.unique(data[3])))

# ========================================================

print("Neural Network")

# Charger/lire les datasets
train = []
train_labels = []
test = []
test_labels = []
for data in datasets:
    train.append(data[0].astype(np.float64))
    train_labels.append(data[1])
    test.append(data[2].astype(np.float64))
    test_labels.append(data[3])

# Initialisation paramètres pour NeuralNet
sizes_iris = [4, 3, 3]
sizes_wine = [11, 8, 2]
sizes_abalone = [8, 6, 3]
sizes_data = [sizes_iris, sizes_wine, sizes_abalone]
nb_iter_iris = 2000
nb_iter_wine = 1000
nb_iter_abalone = 1000
nb_iter_data = [nb_iter_iris, nb_iter_wine, nb_iter_abalone]
threshold_iris = 0.001
threshold_wine = 0.001
threshold_abalone = 0.001
threshold_data = [threshold_iris, threshold_wine, threshold_abalone]
learning_rate_iris = 0.01
learning_rate_wine = 0.01
learning_rate_abalone = 0.01
learning_rate_data = [learning_rate_iris, learning_rate_wine, learning_rate_abalone]

# Initialisation des poids du réseau de neurones
# graphics.courbeApprentissageNN(iris, sizes_iris, "Xavier", 0.01, 20, 1)
# graphics.courbeApprentissageNN(iris, sizes_iris, "Zero", 0.01, 20, 1)

# graphics.courbeApprentissageNN(wine, sizes_wine, "Xavier", 0.001,  5, 75)
# graphics.courbeApprentissageNN(wine, sizes_wine, "Zero", 0.001,  5, 75)

# graphics.courbeApprentissageNN(abalone, sizes_abalone, "Xavier", 0.001, 5, 125)
# graphics.courbeApprentissageNN(abalone, sizes_abalone, "Zero", 0.001, 5, 125)

for index, data in enumerate(datasets):
    print(index)
    print("Dataset", {0:"iris", 1:"wine", 2:"abalone"}[index])
    # Initialisation du NeuralNet
    neuralNet = NeuralNet.NeuralNet(sizes_data[index], "Xavier")

    # Entrainement du NeuralNet
    t1 = time.time()
    neuralNet.train(train[index], train_labels[index], learning_rate_data[index], nb_iter_data[index], threshold_data[index])
    t2 = time.time()

    # Tester le NeuralNet
    t3 = time.time()
    print("Evaluation:")
    print("Train")
    print(neuralNet.evaluate(train[index], train_labels[index]))
    print("Test")
    print(neuralNet.evaluate(test[index], test_labels[index]))
    t4 = time.time()

    print("Temps d'entrainement: {0}\nTemps d'évaluation: {1}\n".format((t2-t1), (t4-t3)))

# Recherche d'hyperparamètres

nb_neurones_data = [[2,3,4,5,6],[4,5,6,7,8],[3,4,5,6,7]]

for idx,data in enumerate(datasets):
    k = 10
    fold_size = int(len(data[1])/k)
    rng = np.random.default_rng(2004)
    data_index = np.arange(len(data[1]))
    rng.shuffle(data_index)
    folds_idx = np.array_split(data_index, k)
    X = train[idx]
    y = train_labels[idx]

    # Choix du nombre de neurones dans la couche cachée
    nb_neurones = nb_neurones_data[idx]
    neurone_error = np.zeros(len(nb_neurones))

    for idx_neurone in range(len(nb_neurones)):
        means = np.zeros(k)
        for fold in range(k):
            fold_train = np.delete(X, folds_idx[fold], axis=0)
            fold_train_labels = np.delete(y, folds_idx[fold], axis=0)
            fold_test = X[folds_idx[fold]]
            fold_test_labels = y[folds_idx[fold]]

            clf = MLPClassifier(hidden_layer_sizes=nb_neurones[idx_neurone], activation="logistic", max_iter=1500, random_state=206).fit(fold_train, fold_train_labels)

            pred = clf.predict(fold_test)

            count = 0
            for index in range(len(fold_test_labels)):
                if pred[index] != fold_test_labels[index]:
                    count += 1

            means[fold] = count / len(fold_test_labels)
            #print(means)
        neurone_error[idx_neurone] = means.mean()
    nb_neurone_data = nb_neurones[np.argmin(neurone_error)]
    plt.plot(nb_neurones, neurone_error)
    plt.xlabel('Nombre de neurones dans la couche cachée')
    plt.ylabel('Erreur Moyenne')
    plt.show()

    # Choix du nombre de couches cachées
    nb_couches= np.array([1,2,3,4,5])
    hidden_layers = []
    for index in range(1, len(nb_couches)+1):
        hidden_layers.append((nb_neurone_data,)*index)

    couche_error = np.zeros(len(nb_couches))

    for idx_couche in range(len(nb_couches)):
        means = np.zeros(k)
        for fold in range(k):
            fold_train = np.delete(X, folds_idx[fold], axis=0)
            fold_train_labels = np.delete(y, folds_idx[fold], axis=0)
            fold_test = X[folds_idx[fold]]
            fold_test_labels = y[folds_idx[fold]]

            clf = MLPClassifier(hidden_layer_sizes=hidden_layers[idx_couche], activation="logistic", max_iter=1500, random_state=508).fit(fold_train, fold_train_labels)

            pred = clf.predict(fold_test)

            count = 0
            for index in range(len(fold_test_labels)):
                if pred[index] != fold_test_labels[index]:
                    count += 1

            means[fold] = count / len(fold_test_labels)
            #print(means)
        couche_error[idx_couche] = means.mean()
    nb_couche_data = nb_couches[np.argmin(couche_error)]
    plt.plot(nb_couches, couche_error)
    plt.xlabel('Nombre de couches cachées')
    plt.ylabel('Erreur Moyenne')
    plt.show()
    print("On utilise", nb_neurone_data, "neurones avec", nb_couche_data, "couches cachées.")

# Entraînement et test

datasets = [iris, wine, abalone]
nb_neurones = [4,7,6]
nb_couches = [1,1,2]
hidden_layers_data = []
for index in range(len(datasets)):
    hidden_layers_data.append((nb_neurones[index],)*nb_couches[index])

sklearn_NN_clf = [MLPClassifier(hidden_layer_sizes=h, activation="logistic", max_iter=1500) for h in hidden_layers_data]

for index, data in enumerate(datasets):
    sklearn_NN_clf[index].fit(data[0].astype(np.float64), data[1].astype(np.float64))

for index, data in enumerate(datasets):
    scores = dict()
    y_pred = sklearn_NN_clf[index].predict(data[2].astype(np.float64))
    y_true = data[3].astype(np.float64)
    scores["accuracy"] = accuracy_score(y_true, y_pred)
    scores["precision"] = precision_score(y_true, y_pred, average="macro")
    scores["recall"] = recall_score(y_true, y_pred, average="macro")
    scores["f1_score"] = f1_score(y_true, y_pred, average="macro")
    print(multilabel_confusion_matrix(y_true, y_pred), "\n", scores, "\n")
