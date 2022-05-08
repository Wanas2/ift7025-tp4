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

    def __init__(self, sizes, init_weights_method="Xavier"):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations

        sizes : Liste de nombre de noeuds dans la couche d'entrée, 
                dans la couche cachée et dans la couche de sortie.
        """

        self.n_inputs = sizes[0]
        self.n_hidden = sizes[1]
        self.n_outputs = sizes[2]
        
        self.cache = {} # Intermediate values

        self.weights_bias = {
            "Weights Hidden" : self._init_weights(self.n_inputs, self.n_hidden, init_weights_method),
            "Bias Hidden" : self._init_bias(self.n_inputs, self.n_hidden, init_weights_method),
            "Weights Output" : self._init_weights(self.n_hidden, self.n_outputs, init_weights_method),
            "Bias Output" : self._init_bias(self.n_hidden, self.n_outputs, init_weights_method)
        }
        
        
        
        
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
        self.iter = n_iterations
        train_labels = encoding.transformLabel(train_labels)
        
        for i in range(self.iter):
            # Batch
            x = train
            y = train_labels
            
            # Forward propagation
            output = self._forward_propagation(x)
            # Backpropagation
            _ = self._backward_propagation(y, output)
            # Weights and bias update
            self.weights_bias_update(learning_rate)

            # error = sum((self.cache["A Output"]-y)**2)/2
            # if error < threshold:
            #     break



    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        return self._forward_propagation(x)


        
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
        labels = np.unique(y)

        predictions = list()
        for x in X:
            predictions.append(self.predict(x))

        predictions = np.array(predictions)

        unique_labels = set(labels)
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

    def _init_weights(self, input_size, output_size, init_method):
        if init_method == "Zero":
            weights = np.zeros((output_size, input_size)) 
        if init_method == "Xavier":
            weights = np.random.randn(output_size, input_size) * np.sqrt(2/(input_size+output_size))
        return weights

    def _init_bias(self, input_size, output_size, init_method):
        if init_method == "Zero":
            weights = np.zeros((output_size, 1))
        if init_method == "Xavier":
            weights = np.zeros((output_size, 1)) * np.sqrt(1/(input_size))
        return weights


    def _forward_propagation(self, x):
        """
        Faire la propagation à partir d'un exemple x donné en entrée.
        Exemple est de taille 1xm.
        """
        self.cache["X"] = x
        self.cache["Z Hidden"] = np.matmul(self.weights_bias["Weights Hidden"], self.cache["X"].T) + self.weights_bias["Bias Hidden"]
        self.cache["A Hidden"] = maths.sigmoid(self.cache["Z Hidden"])
        self.cache["Z Output"] = np.matmul(self.weights_bias["Weights Output"], self.cache["A Hidden"]) + self.weights_bias["Bias Output"]
        self.cache["A Output"] = maths.sigmoid(self.cache["Z Output"])
        return self.cache["A Output"]

    def _backward_propagation(self, y, output):
        n_labels = y.shape[0]
        
        dZOutput = output - y.T
        dWOutput = (1/n_labels) * np.matmul(dZOutput, self.cache["A Hidden"].T)
        dbOutput = (1/n_labels) * np.sum(dZOutput, axis=1, keepdims=True)

        dAHidden = np.matmul(self.weights_bias["Weights Output"].T, dZOutput)
        dZHidden = dAHidden * maths.sigmoid_prime(self.cache["Z Hidden"])
        dWHidden = (1/n_labels) * np.matmul(dZHidden, self.cache["X"])
        dbHidden = (1/n_labels) * np.sum(dZHidden, axis=1, keepdims=True)

        self.grads = {"Weights Hidden": dWHidden, "Bias Hidden": dbHidden, "Weights Output": dWOutput, "Bias Output": dbOutput}
        return self.grads
    
    def weights_bias_update(self, learning_rate):
        for key in self.weights_bias:
            self.weights_bias[key] = self.weights_bias[key] - learning_rate * self.grads[key]