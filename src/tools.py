import matplotlib.pyplot as plt
from random import shuffle
import numpy as np


def plot_loss(iterations, train_losses, test_losses, x_label, y_label, title=None, filename=None, to_save=False):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)

    plt.plot(iterations, train_losses, label='Train')
    plt.plot(iterations, test_losses, label='Test')
    plt.xticks(np.arange(min(iterations), max(iterations) + 1, 10))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='best')

    if to_save:
        fig.savefig("../logs/" + filename + ".png")
    else:
        plt.show()
    # plt.close(fig)


def preprocess(data, N):
    # Sample N observations from the train data
    shuffle(data["Xtrain"])
    shuffle(data["ytrain"])

    if N is not None :
        data["Xtrain"] = data["Xtrain"][:N]
        data["ytrain"] = data["ytrain"][:N]

    # Rescale y to binary
    data["ytrain"] = [data["ytrain"][i, 0]
                      for i in range(data["ytrain"].shape[0])]
    data["ytest"] = [data["ytest"][i, 0]
                     for i in range(data["ytest"].shape[0])]

    return data


def plot_data(X, y):
    X_1 = [X[i][0] for i in range(len(X))]
    X_2 = [X[i][1] for i in range(len(X))]

    plt.scatter(X_1, X_2, c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.show()