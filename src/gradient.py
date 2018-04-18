import math
import numpy as np


class Gradient :
    def __init__(self, param):
        self.data_generator = self.initialize_data_generator(data=param["data"],
                                                  batch_size=param["batch_size"])

        self.n_samples = len(param["data"]["Xtrain"])
        self.batch_size = param["batch_size"]
        self.stepsize_type = param["stepsize_type"]
        self.initial_stepsize = param["initial_stepsize"]

        # SAG parameters
        self.seen_gradients_counts = 0
        self.stored_gradients = [0] * self.n_samples
        self.first_seen = [0] * self.n_samples

    @staticmethod
    def initialize_data_generator(data, batch_size):
        X_data = data["Xtrain"]
        y_data = data["ytrain"]
        n_samples = len(X_data)

        while True:
            sample_idxs = list(np.random.choice(n_samples, batch_size))
            X_mini_batch = [np.array(X_data[i]) for i in sample_idxs]
            y_mini_batch = [y_data[i] for i in sample_idxs]
            yield X_mini_batch, y_mini_batch, sample_idxs

    @staticmethod
    def stepsize(stepsize_type, initial_stepsize, it):
        if stepsize_type == "constant" :
            return initial_stepsize
        if stepsize_type == "sqrt" :
            return 1 / math.sqrt(it + 1)

    @staticmethod
    def compute_single_gradient(model, X, y_true):
        y_pred = model.predict(X)

        if model.type == "square":
            gradient = (y_pred - y_true) * y_pred * (1 - y_pred) * X
        if model.type == "logistic":
            gradient = (- y_true * (1 - y_pred) + (1 - y_true) * y_pred) * X
        return gradient

    def compute_gradient(self, model, optimizer):
        X, y, index = next(self.data_generator)
        gradients = np.array([self.compute_single_gradient(model, X[i], y[i])
                     for i in range(self.batch_size)])
        return np.sum(gradients, axis=0) / self.batch_size

    def sgd_step(self, model, it):
        gradient = self.compute_gradient(model)
        step_size = self.stepsize(self.stepsize_type, self.initial_stepsize, it)
        model.theta -= step_size * gradient

    def update_gradients(self, gradient, index):
        if self.first_seen[index] == 0 :
            self.first_seen[index] = 1
            self.seen_gradients_counts += 1
        self.stored_gradients[index] = gradient

    def compute_sag_direction(self):
        return np.sum(self.stored_gradients, axis=0) \
                       / self.seen_gradients_counts

    def compute_sag_scalar(self, model, X, y):
        if model.type == "square" :


    def sag_step(self, model, it):
        X, y, index = next(self.data_generator)
        gradient = self.compute_single_gradient(model, X[0], y[0])
        self.update_gradients(gradient, index)
        sag_direction = self.compute_sag_direction(gradient)
        step_size = self.stepsize(self.stepsize_type, self.initial_stepsize, it)
        model.theta -= step_size * sag_direction


