import numpy as np
from tqdm import tqdm


class LogisticRegression:

    def __init__(self, type, gradient_stepsize, d=100, theta=None):
        if theta is None:
            self.theta = np.random.rand(d) * 2 - 1
        else:
            self.theta = theta

        self.type = type
        self.gradient_stepsize = gradient_stepsize

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(- z))

    def error(self, X, y_true):
        N = len(y_true)
        return sum([self.single_error(X[i], y_true[i])
                    for i in range(N)]) / N

    def single_error(self, X, y_true):
        y_pred = round(self.predict(X))
        return abs(y_true - y_pred)

    def loss(self, X, y_true):
        N = len(y_true)
        return sum([self.single_loss(X[i], y_true[i])
                    for i in range(N)]) / N

    def single_loss(self, X, y_true):
        y_pred = self.predict(X)
        if self.type == "square":
            return (y_pred - y_true) ** 2
        if self.type == "logistic":
            return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.theta))

    def compute_gradient(self, X, y_true):
        # TODO : Check the gradient
        y_pred = self.predict(X)

        if self.type == "square":
            gradient = (y_pred - y_true) * y_pred * (1 - y_pred) * X
        if self.type == "logistic":
            gradient = (- y_true * (1 - y_pred) + (1 - y_true) * y_pred) * X
        return gradient

    def compute_batch_gradient(self, X, y_true):
        N = X.shape[0]
        gradients = [self.compute_gradient(X[i], y_true[i])
                     for i in range(N)]
        return sum(gradients) / N

    def gradient_step(self, X, y_true):
        gradient = self.compute_batch_gradient(X, y_true)
        self.theta -= self.gradient_stepsize * gradient

    def log(self, log_dict, mat, it):
        log_dict["train_losses"].append(self.loss(X=mat["Xtrain"],
                                                  y_true=mat["ytrain"]))
        log_dict["test_losses"].append(self.loss(X=mat["Xtest"],
                                                 y_true=mat["ytest"]))
        log_dict["iterations"].append(it)
        log_dict["train_errors"].append(self.error(X=mat["Xtrain"],
                                                   y_true=mat["ytrain"]))
        log_dict["test_errors"].append(self.error(X=mat["Xtest"],
                                                  y_true=mat["ytest"]))

    def run_gradient_descent(self, mat, n_iter, log_freq):
        log_dict = {"train_losses": [],
                    "test_losses": [],
                    "iterations": [],
                    "train_errors": [],
                    "test_errors": []}

        for it in tqdm(range(n_iter)):
            self.gradient_step(X=mat["Xtrain"], y_true=mat["ytrain"])

            if it % log_freq == 0:
                self.log(log_dict, mat, it)

        return log_dict
