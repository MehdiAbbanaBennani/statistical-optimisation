import matplotlib.pyplot as plt
from random import shuffle


def plot_loss(iterations, train_losses, test_losses, y_label, title=None, filename=None, to_save=False):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)

    plt.plot(iterations, train_losses, label='Train')
    plt.plot(iterations, test_losses, label='Test')

    ax.set_xlabel('Iterations')
    ax.set_ylabel(y_label)
    ax.legend(loc='best')

    if to_save:
        fig.savefig("logs/" + filename + ".png")
    else:
        plt.show()
    # plt.close(fig)


def preprocess(data, N):
    # Sample N observations from the train data
    shuffle(data["Xtrain"])
    shuffle(data["ytrain"])

    data["Xtrain"] = data["Xtrain"][:N]
    data["ytrain"] = data["ytrain"][:N]

    # Rescale y to binary
    data["ytrain"] = [(data["ytrain"][i, 0] + 1) / 2
                      for i in range(data["ytrain"].shape[0])]
    data["ytest"] = [(data["ytest"][i, 0] + 1) / 2
                     for i in range(data["ytest"].shape[0])]

    return data
