import numpy as np

def transformLabel(labels):
	longueur = len(labels)
	classes = np.unique(labels)
	nb_classes = len(classes)
	float_labels = labels.astype(np.float64)

	transformed_labels = np.zeros((longueur, nb_classes))
	for idx_labels in range(longueur):
		for idx_classes in range(nb_classes):
			if float_labels[idx_labels] == idx_classes:
				transformed_labels[idx_labels,idx_classes] = 1
	return transformed_labels

def transform_outputs(outputs):
    max_index = outputs.tolist().index(max(outputs))

    for index in range(len(outputs)):
        if index == max_index:
            outputs[index] = 1
        else:
            outputs[index] = 0