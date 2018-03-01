import matplotlib.pyplot as plt


def plot(x, func):
    """
    Plots func over the vector x
    :param x: a list
    :param func: a functions of x which returns a scalar
    """
    fig = plt.figure(figsize=(10, 5))
    y = [func(x_i) for x_i in x]

    plt.plot(x, y, label='Debug plot')
    plt.show()
    plt.close(fig)