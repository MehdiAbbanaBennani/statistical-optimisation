import numpy as np
from tqdm import tqdm
from gradient import Gradient


class LogisticRegression:

    def __init__(self, type, mu, gradient_param, data, d=100, theta=None):
        if theta is None:
            self.theta = np.random.rand(d) * 2 - 1
        else:
            self.theta = theta

        self.type = type
        self.gradient = Gradient(gradient_param)
        self.mat = data
        self.n_samples = data["Xtrain"].shape[0]
        self.mu = mu

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(- z))

    def error(self, X, y_true):
        N = len(y_true)
        return sum([self.single_error(X[i], y_true[i])
                    for i in range(N)]) / N

    def single_error(self, X, y_true):
        # y_pred = round(self.predict(X))
        y_pred = self.predict_label(X)
        return abs(y_true - y_pred) / 2

    def loss(self, X, y_true):
        N = len(y_true)
        return sum([self.single_loss(X[i], y_true[i])
                    for i in range(N)]) / N

    def single_loss(self, X, y_true):
        y_pred = self.predict(X)
        if self.type == "square":
            return (y_pred - y_true) ** 2
        if self.type == "logistic":
            return np.log(1 + np.exp(- y_true * y_pred))
            # return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def predict(self, X):
        # return self.sigmoid(np.dot(X, self.theta))
        return np.dot(X, self.theta)

    def predict_label(self, X):
        y_pred = self.predict(X)
        if y_pred < 0 :
            return -1
        else :
            return 1

    def log(self, log_dict, it, log_freq):
        log_dict["train_losses"].append(self.loss(X=self.mat["Xtrain"],
                                                  y_true=self.mat["ytrain"]))
        log_dict["test_losses"].append(self.loss(X=self.mat["Xtest"],
                                                 y_true=self.mat["ytest"]))
        log_dict["train_errors"].append(self.error(X=self.mat["Xtrain"],
                                                   y_true=self.mat["ytrain"]))
        log_dict["test_errors"].append(self.error(X=self.mat["Xtest"],
                                                  y_true=self.mat["ytest"]))
        if log_freq == "epoch" :
            log_dict["iterations"].append(it / self.n_samples)
        else :
            log_dict["iterations"].append(it)

    def compute_n_iter(self, n_epoch):
        return n_epoch * (self.n_samples // self.gradient.batch_size)

    def log_freq_to_iter(self, log_freq):
        if log_freq == "epoch" :
            return self.n_samples
        else :
            return log_freq

    def run_optimizer(self, n_epoch, log_freq, optimizer):
        log_dict = {"train_losses": [],
                    "test_losses": [],
                    "iterations": [],
                    "train_errors": [],
                    "test_errors": []}
        n_iter = self.compute_n_iter(n_epoch)

        for it in tqdm(range(n_iter)):
            if optimizer == "sgd" :
                self.gradient.sgd_step(model=self, it=it)
            if optimizer == "sag":
                self.gradient.sag_step(model=self, it=it)

            if it % self.log_freq_to_iter(log_freq) == 0:
                self.log(log_dict, it, log_freq)

        return log_dict