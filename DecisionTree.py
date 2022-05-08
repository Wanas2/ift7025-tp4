import numpy as np

import utils.graphics as graphics
import utils.metrics as metrics
import utils.selections as selections
import utils.maths as maths

class DecisionTreeClassifier: #nom de la class à changer

    def __init__(self, max_depth=None):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.tree = {}	
        self.max_depth = max_depth
        self.depth = 0

    def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le nombre d'attributs (le nombre de caractéristiques)
        
        train_labels : est une matrice numpy de taille nx1
        
        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        
        """
        self.labels_ = np.unique(train_labels)
        self.n_labels_ = len(self.labels_)

        self.features_ = list(range(train.shape[1]))

        self.tree = self._decision_tree_learning(train, train_labels, 1)
        
    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        return self._predict(self.tree, x)
        
    def evaluate(self, X, y):
        """
        c'est la méthode qui va évaluer votre modèle sur les données X
        l'argument X est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple de test dans le dataset
        m : le nombre d'attributs (le nombre de caractéristiques)
        
        y : est une matrice numpy de taille nx1
        
        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        """
        predictions = list()
        for x in X:
            predictions.append(self.predict(x))

        predictions = np.array(predictions)

        unique_labels = set(self.labels_)
        unique_labels.update(y)
        self.unique_labels = list(unique_labels)

        self.n_labels = len(unique_labels)
        
        return self._get_metrics(predictions, y)
        
    
    # Vous pouvez rajouter d'autres méthodes et fonctions,
    # il suffit juste de les commenter.    
    def _decision_tree_learning(self, train, train_labels, depth):
        if len(np.unique(train_labels)) == 1:
            return train_labels[0]

        m = self._majority_value(list(train_labels))
        if self.max_depth and depth >= self.max_depth:
            return m

        best_feature, best_split = self._choose_best_feature_with_best_split(train, train_labels)
        
        tree = dict()
        tree["feature"] = best_feature
        tree["value"] = best_split["value"]
        tree["sample"] = self._sample_distribution(train_labels)

        lset, rset = best_split["split"]
        
        if depth > self.depth:
            self.depth += 1
        
        tree["left"] = self._decision_tree_learning(lset[0], lset[1], depth+1)
        tree["right"] = self._decision_tree_learning(rset[0], rset[1], depth+1)

        return tree 

    def _choose_best_feature_with_best_split(self, train, train_labels):
        _feature, _gain = -np.inf, -np.inf
        _split = dict()
        
        for feature in self.features_:
            out = self._best_split(train, train_labels, feature)
            gain = out["gain"]
            if gain > _gain:
                _feature, _gain = feature, gain
                del out["gain"]
                _split = out

        return _feature, _split

    def _best_split(self, train, train_labels, feature):
        _value, _gain = -np.inf, -np.inf
        _split = tuple()

        for value in np.unique(train):
            ltrain, ltrain_labels, rtrain, rtrain_labels = self._get_split(train, train_labels, feature, value)

            gain = maths.entropy(train_labels)
            gain -= (len(ltrain_labels) / len(train_labels)) * maths.entropy(ltrain_labels) 
            gain -= (len(rtrain_labels) / len(train_labels)) * maths.entropy(rtrain_labels)
            
            if gain > _gain:
                _value, _gain = value, gain
                _split = (ltrain, ltrain_labels), (rtrain, rtrain_labels)

        return {"value": _value, "gain": _gain, "split": _split}

    def _get_split(self, X, y, feature, value):
        inf = X[:, feature] < value
        sup = X[:, feature] >= value
        return X[inf], y[inf], X[sup], y[sup]

    def _majority_value(self, labels):
        return max(set(labels), key=labels.count)

    def _predict(self, tree, x):
        if self.n_labels_ == 1:
            return self.labels_[0]
        
        if x[tree['feature']] < tree['value']:
            if isinstance(tree['left'], dict):
                return self._predict(tree['left'], x)

            return tree['left']

        if isinstance(tree['right'], dict):
            return self._predict(tree['right'], x)

        return tree['right']

    def _get_metrics(self, y_pred, y_true):
        if self.n_labels == 2:
            con_matrix = metrics.binary_confusion_matrix(y_pred, y_true, self.unique_labels[0])
            accuracy = metrics.accuracy_metrics(con_matrix)
            precision = metrics.precision_metrics(con_matrix)
            recall = metrics.recall_metrics(con_matrix)
            f1_score = metrics.f1_score_metrics(con_matrix)
        elif self.n_labels > 2:
            con_matrix = metrics.multilabel_confusion_matrix(y_pred, y_true, self.unique_labels)
            accuracy = [metrics.accuracy_metrics(con_matrix[i]) for i in range(self.n_labels)]
            precision = [metrics.precision_metrics(con_matrix[i]) for i in range(self.n_labels)] 
            recall = [metrics.recall_metrics(con_matrix[i]) for i in range(self.n_labels)]
            f1_score = [metrics.f1_score_metrics(con_matrix[i]) for i in range(self.n_labels)]

        return ({
            "con_matrix": con_matrix,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        })
        
    # ===========================         
    def _sample_distribution(self, labels):
        return np.unique(labels, return_counts=True)[1].tolist()

    def _get_error(self, label_pred, test_labels):
        error = 0
        for i in range(len(test_labels)):
            if label_pred != test_labels[i]:
                error += 1
        return float(error)

    def prune(self, tree, test, test_labels, default=None):
        if len(test_labels) == 0:
            return default

        ltest, ltest_labels = [], []
        rtest, rtest_labels = [], []
        if (isinstance(tree['right'], dict) or isinstance(tree['left'], dict)):
            ltest, ltest_labels, rtest, rtest_labels = self._get_split(test, test_labels, tree["feature"], tree["value"])

        if isinstance(tree['left'], dict):
            default = self._majority_value(tree["sample"])
            tree['left'] = self.prune(tree['left'], ltest, ltest_labels, default)

        if isinstance(tree['right'], dict):
            default = self._majority_value(tree["sample"])
            tree['right'] = self.prune(tree['right'], rtest, rtest_labels, default)

        if not isinstance(tree['left'], dict) and not isinstance(tree['right'], dict):
            ltest, ltest_labels, rtest, rtest_labels = self._get_split(test, test_labels, tree["feature"], tree["value"])
            
            lerror = self._get_error(tree['left'], ltest_labels)
            rerror = self._get_error(tree['right'], rtest_labels)

            label_pred = self._majority_value(list(test_labels))
            merror = self._get_error(label_pred, test_labels)

            if merror <= lerror + rerror:
                return label_pred
            
        return tree

    def count_leaves(self, tree):
        if not isinstance(tree, dict):
            return 1
        return self.count_leaves(tree["left"]) + self.count_leaves(tree["right"])
    
    def get_depth(self, tree):
        if not isinstance(tree, dict):
            return 0

        if tree["left"]:
            hl = self.get_depth(tree["left"])
        else:
            hl = 0

        if tree["right"]:
            hr = self.get_depth(tree["right"])
        else:
            hr = 0

        return 1 + max(hl, hr)

    def print_tree(self, tree, depth):
        """
        Prints the whole tree from the current node to the bottom
        """
        if isinstance(tree, dict):
            print("\t" * depth, "|----", "Feature: ", tree["feature"], ", Value: ", tree["value"]) 
        
            
            if isinstance(tree["left"], dict): 
                self.print_tree(tree["left"], depth+1)
            else:
                print("\t" * (depth+1), "|---- Leaf: ", tree["left"])
            
            if isinstance(tree["right"], dict): 
                self.print_tree(tree["right"], depth+1)
            else:
                print("\t" * (depth+1), "|---- Leaf: ", tree["right"])
        else:
            print("\t" * depth, "|---- Leaf: ", tree)
