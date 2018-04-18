import numpy as np


class DataGenerator :
  def __init__(self, x_mu, x_sigma, x_dim, noise_dev):
    self.mu = x_mu
    self.sigma = x_sigma
    self.dim = x_dim
    self.noise_dev = noise_dev

  @staticmethod
  def y_function(x):
    # new_x = [x_i**2 for x_i in x]
    if sum(x) >= 0 :
      return 1
    else :
      return 0

  def add_noise(self, x):
    return [x_i + np.random.normal(scale=self.noise_dev) for x_i in x]

  def generate_sample(self):
    x = [np.random.normal(self.mu[i], self.sigma[i]) for i in range(self.dim)]
    y = self.y_function(x)
    x = self.add_noise(x)
    return (x, y)

  def generate(self, n_samples):
    data = [self.generate_sample() for _ in range(n_samples)]
    X = [data[i][0] for i in range(n_samples)]
    y = [data[i][1] for i in range(n_samples)]
    return X, y

  @staticmethod
  def preprocess(X, y, split_ratio=0.9):
    data = {}
    n_samples = len(X)
    n_train_samples = int(split_ratio * n_samples)

    data["Xtrain"] = X[:n_train_samples]
    data["ytrain"] = y[:n_train_samples]
    data["Xtest"] = X[(n_train_samples + 1):]
    data["ytest"] = y[(n_train_samples + 1):]
    return data