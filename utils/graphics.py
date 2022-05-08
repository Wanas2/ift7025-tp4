import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_tree(tree):
    ax = plt.gca()
    ax.set_axis_off()
    
    build_tree(tree, 0.25, 0.25)

    ax.set_xlim(-1, 2)
    ax.set_ylim(-2, 1)

    plt.show()

def build_tree(tree, left, bottom, depth=0, ax=plt.gca()):
    height = width = 0.5
    right = left + width
    top = bottom + height

    p = patches.Rectangle((left, bottom), width, height, fill=False)
    ax.add_artist(p)

    if isinstance(tree, dict):
        ax.text(
            0.5*(left+right), 
            0.5*(bottom+top+0.1), 
            'X[{0}] < {1}'.format(tree["feature"], tree["value"]),
            horizontalalignment='center', 
            verticalalignment='center',
            fontsize=8
        )

        ax.text(
            0.5*(left+right), 
            0.5*(bottom+top-0.1), 
            '{0}'.format(tree["sample"]),
            horizontalalignment='center', 
            verticalalignment='center',
            fontsize=8
        )

        xy1 = (0.5*(left+right), bottom)

        ha = 0.2
        y2 = bottom-ha

        ax.annotate("", xy=xy1, xytext=(left, y2), arrowprops=dict(arrowstyle="<-"))
        ax.annotate("", xy=xy1, xytext=(right, y2), arrowprops=dict(arrowstyle="<-"))

        bottom = bottom - height - ha
        build_tree(tree["left"], left - 0.6*width, bottom)
        build_tree(tree["right"], right - 0.4*width, bottom)
    else:
        ax.text(
            0.5*(left+right), 
            0.5*(bottom+top), 
            '{0}'.format(tree),
            horizontalalignment='center', 
            verticalalignment='center',
            fontsize=8
        )

def plot_learning_curve(data, clf, start_size=1, end_size=None, num_sample=20):
    _data = np.concatenate((
                np.copy(data[0].astype(np.float64)), 
                np.copy(data[1])[np.newaxis, :].T
                ), axis=1
            )

    accuracies = list()

    if not end_size:
        end_size = len(_data)-1

    for size in range(start_size, end_size):
        mean = 0
        for i in range(num_sample):
            np.random.shuffle(_data)
            train, train_labels = np.copy(_data[:size+1, :-1]), np.copy(_data[:size+1, -1])
            test, test_labels = np.copy(_data[size+1:, :-1]), np.copy(_data[size+1:, -1])

            clf.train(train, train_labels)

            count = 0
            for index in range(len(test_labels)):
                if clf.predict(test[index]) == test_labels[index]:
                    count += 1

            mean += count / len(test_labels)
        mean /= num_sample        
        accuracies.append(mean)

    plt.plot(range(start_size, end_size), accuracies)

    plt.xlabel("Training set size")
    plt.ylabel("Proportion correct on test set")

    plt.show()

def courbeApprentissageNN(dataset, sizes_dataset, init_weights, learning_rate, nb_iter, step):
    import NeuralNet
    
    data_ = np.concatenate((np.copy(dataset[0].astype(np.float64)), np.copy(dataset[1])[np.newaxis, :].T), axis=1)

    accuracies = list()
    for size in range(1, len(data_)-1, step):
        print(size)
        mean = 0
        for i in range(20):
            np.random.shuffle(data_)
            train, train_labels = np.copy(data_[:size+1, :-1]), np.copy(data_[:size+1, -1])
            test, test_labels = np.copy(data_[size+1:, :-1]), np.copy(data_[size+1:, -1])

            # print(train_labels)

            NNet = NeuralNet.NeuralNet(sizes_dataset,init_weights)
            NNet.train(train, train_labels, learning_rate, nb_iter)
            # print(dT.tree)

            count = 0
            for index in range(len(test_labels)):
                if NNet.predict(test[index]) == test_labels[index]:
                    count += 1

            mean += count / len(test_labels)
        mean /= 20        
        accuracies.append(mean)

    plt.plot(range(1, len(data_)-1, step), accuracies)
    plt.xlabel('Training set size')
    plt.ylabel('Proportion correct on test set')
    plt.show()
