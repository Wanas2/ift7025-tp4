import numpy as np

def euclidian_distance(x1, x2):
    distance = 0.0
    for i in range(len(x1)-1):
        distance += (x1[i] - x2[i])**2
    return  np.sqrt(distance)

def entropy(X):
    probs = np.unique(X, return_counts=True)[1] / len(X)
    return sum([-p_i * np.log2(p_i) for p_i in probs if p_i > 0])

def sigmoid(y):
	"""
	Fontion d'activation sigmoïd.
	"""
	return 1.0/(1.0+np.exp(-y))
		
def sigmoid_prime(y):
	"""
	Dérivé de la fonction d'activation sigmoïd.
	"""
	return sigmoid(y)*(1-sigmoid(y))
