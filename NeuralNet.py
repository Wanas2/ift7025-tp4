"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
    * train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
    * predict 	: pour prédire la classe d'un exemple donné.
    * evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np
import utils.graphics as graphics
import utils.metrics as metrics
import utils.selections as selections
import utils.maths as maths
import utils.encoding as encoding

# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class NeuralNet: #nom de la class à changer

    def __init__(self, sizes, init_weights):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations

        sizes : Liste de nombre de noeuds dans la couche d'entrée, 
                dans la couche cachée et dans la couche de sortie.
        """
        self.sizes = sizes
        if init_weights == "Xavier":
            self.weights = [np.random.randn(y,x) * np.sqrt(2/(x+y)) for x,y in zip(sizes[:-1], sizes[1:])]
        if init_weights == "Zero":
            self.weights = [np.zeros((y,x)) for x,y in zip(sizes[:-1], sizes[1:])]
        
        
    def train(self, train, train_labels, learning_rate, n_iterations, threshold): #vous pouvez rajouter d'autres attributs au besoin
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
        train_labels = encoding.transformLabel(train_labels)
        a_input = np.zeros(self.sizes[0])
        a_hidden = np.zeros(self.sizes[1])
        input_hidden = np.zeros(self.sizes[1])
        delta_hidden = np.zeros(self.sizes[1])
        a_output = np.zeros(self.sizes[2])
        input_output = np.zeros(self.sizes[2])
        delta_output = np.zeros(self.sizes[2])
        

        count=0
        for iter in range(n_iterations):
            for x,y in zip(train,train_labels):
                # Forward propagation
                # a_input = np.copy(x)
                for idx in np.arange(self.sizes[0]):
                    a_input[idx] = x[idx]

                for idx in np.arange(self.sizes[1]):
                    input_hidden[idx] = sum(self.weights[0][idx]*a_input)
                    a_hidden[idx] = maths.sigmoid(input_hidden[idx])
                for idx in np.arange(self.sizes[2]):
                    input_output[idx] = sum(self.weights[1][idx]*a_hidden)
                    a_output[idx] = maths.sigmoid(input_output[idx])

                # Backpropagation
                #Output layer
                for idx in np.arange(self.sizes[2]):
                    delta_output[idx] = maths.sigmoid_prime(input_output[idx]) * (y[idx] - a_output[idx])
                
                #Hidden layer
                for idx in np.arange(self.sizes[1]):
                    for idx_out in np.arange(self.n_labels_):
                        delta_hidden[idx] = maths.sigmoid_prime(input_hidden[idx]) * sum(self.weights[1][idx_out][idx] * delta_output)
                
                # Mise à jour des poids
                for idx_cache in np.arange(self.sizes[1]):
                    for idx in np.arange(self.sizes[0]):
                        self.weights[0][idx_cache][idx] += learning_rate * delta_hidden[idx_cache] * a_input[idx]
                
                for idx_output in np.arange(self.sizes[2]):
                    for idx in np.arange(self.sizes[1]):
                        self.weights[1][idx_output][idx] += learning_rate * delta_output[idx_output] * a_hidden[idx]
                
            error = sum((a_output-y)**2)/2
            if error < threshold:
                break
            count+=1
        print("count", count)


    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        a_input = np.zeros(self.sizes[0])
        a_cache = np.zeros(self.sizes[1])
        inp_cache = np.zeros(self.sizes[1])
        a_output = np.zeros(self.sizes[2])
        inp_output = np.zeros(self.sizes[2])

        for idx in np.arange(self.sizes[0]):
            a_input[idx] = x[idx]
        # a_input = np.copy(x)
        for idx in np.arange(self.sizes[1]):
            inp_cache[idx] = sum(self.weights[0][idx]*a_input)
            a_cache[idx] = maths.sigmoid(inp_cache[idx])
        for idx in np.arange(self.sizes[2]):
            inp_output[idx] = sum(self.weights[1][idx]*a_cache)
            a_output[idx] = maths.sigmoid(inp_output[idx])


        max_out = 0
        for idx in np.arange(self.sizes[2]):
            if a_output[idx] > max_out:
                max_out = a_output[idx]

        
        for idx in np.arange(len(self.labels_)):
            if max_out == a_output[idx]:
                return self.labels_[idx]

            
        
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
        # predictions = np.zeros(len(X), dtype='<U20')
        # for idx, x in enumerate(X):
        # 	predictions[idx] = (self.predict(x))

        # return self._metrics(predictions, y)
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


        


