import numpy as np
from tqdm import tqdm
from gradient import Gradient


class LogisticRegression:

    def __init__(self, type, gradient_param, data, d=100, theta=None):
        if theta is None:
            self.theta = np.random.rand(d) * 2 - 1
        else:
            self.theta = theta

        self.type = type
        self.gradient = Gradient(gradient_param)
        self.mat = data

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

    def log(self, log_dict, it):
        log_dict["train_losses"].append(self.loss(X=self.mat["Xtrain"],
                                                  y_true=self.mat["ytrain"]))
        log_dict["test_losses"].append(self.loss(X=self.mat["Xtest"],
                                                 y_true=self.mat["ytest"]))
        log_dict["iterations"].append(it)
        log_dict["train_errors"].append(self.error(X=self.mat["Xtrain"],
                                                   y_true=self.mat["ytrain"]))
        log_dict["test_errors"].append(self.error(X=self.mat["Xtest"],
                                                  y_true=self.mat["ytest"]))

    def run_optimizer(self, n_iter, log_freq, optimizer):
        # Stochastic Gradient Descent
        log_dict = {"train_losses": [],
                    "test_losses": [],
                    "iterations": [],
                    "train_errors": [],
                    "test_errors": []}

        for it in tqdm(range(n_iter)):
            if optimizer == "sgd" :
                self.gradient.sgd_step(model=self, it=it)
            if optimizer == "sag":
                self.gradient.sag_step(model=self, it=it)

            if it % log_freq == 0:
                self.log(log_dict, it)

        return log_dict