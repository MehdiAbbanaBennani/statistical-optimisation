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
        self.previous_direction = 0
        self.previous_seen_count = 1
        self.seen_gradients_counts = 1
        self.stored_sag_scalars = [0] * self.n_samples
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

    # @staticmethod
    def compute_single_gradient(self, model, X, y_true):
        # y_pred = model.predict(X)

        # if model.type == "square":
        #     gradient = (y_pred - y_true) * y_pred * (1 - y_pred) * X
        # if model.type == "logistic":
        #     gradient = (- y_true * (1 - y_pred) + (1 - y_true) * y_pred) * X
        return self.compute_sag_scalar(model, X, y_true) * X

    def compute_gradient(self, model):
        X, y, index = next(self.data_generator)
        gradients = np.array([self.compute_single_gradient(model, X[i], y[i])
                     for i in range(self.batch_size)])
        return np.sum(gradients, axis=0) / self.batch_size

    def sgd_step(self, model, it):
        gradient = self.compute_gradient(model)
        step_size = self.stepsize(self.stepsize_type, self.initial_stepsize, it)
        # model.theta -= step_size * gradient
        model.theta -= (step_size * gradient + model.mu * model.theta)

    @staticmethod
    def compute_sag_scalar(model, X, y_true):
        y_pred = model.predict(X)

        if model.type == "square":
            # scalar = (y_pred - y_true) * y_pred * (1 - y_pred)
            scalar = (y_pred - y_true)
        if model.type == "logistic":
            scalar = - y_true * np.exp(- y_true * y_pred) / (1 + np.exp(- y_true * y_pred))
            # scalar = (- y_true * (1 - y_pred) + (1 - y_true) * y_pred)
        return scalar

    def update_sag_scalars(self, sag_scalar, index):
        self.previous_seen_count = self.seen_gradients_counts
        if self.first_seen[index] == 0:
            self.first_seen[index] = 1
            self.seen_gradients_counts += 1
        self.stored_sag_scalars[index] = sag_scalar

    def compute_sag_direction(self, sag_scalar, x, index):
        old_grad = self.previous_direction * self.previous_seen_count
        correction_term = (sag_scalar - self.stored_sag_scalars[index]) * x
        return (old_grad + correction_term) / self.seen_gradients_counts

    def sag_step(self, model, it):
        assert self.batch_size == 1

        X, y, indexes = next(self.data_generator)
        sag_scalar = self.compute_sag_scalar(model, X[0], y[0])
        sag_direction = self.compute_sag_direction(sag_scalar, X[0], indexes[0])
        self.previous_direction = sag_direction
        self.update_sag_scalars(sag_scalar, indexes[0])

        step_size = self.stepsize(self.stepsize_type, self.initial_stepsize, it)
        # model.theta -= step_size * sag_direction
        model.theta -= (step_size * sag_direction + model.mu * model.theta)

