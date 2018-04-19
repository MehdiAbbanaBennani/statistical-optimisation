import os
import sys

sys.path.append(os.getcwd() + "/src")

from datagen import DataGenerator
from logistic_regression import LogisticRegression
from tools import plot_loss, plot_data, preprocess
from scipy.io import loadmat
from sklearn.model_selection import ParameterGrid


def run(optimizer, loss_type, mu):
  # Parameters
  n_samples = 1000
  N = None
  n_epoch = 50
  log_freq = "epoch"
  synthetic_data = False
  d = 100
  gradient_parameters = {"batch_size": 1,
                         "stepsize_type": "constant",
                         "initial_stepsize": 0.0001}
  to_save = True

  # Data generation
  if synthetic_data:
    d = 2
    data_generator = DataGenerator(x_mu=[0] * d,
                                   x_sigma=[1] * d,
                                   x_dim=d,
                                   noise_dev=0.8)
    X, y = data_generator.generate(n_samples)
    data = data_generator.preprocess(X, y)
    plot_data(data["Xtrain"], data["ytrain"])
  else:
    data = preprocess(data=loadmat('../data_orsay_2017.mat'), N=N)

  # Train
  gradient_parameters["data"] = data
  logistic_model = LogisticRegression(type=loss_type,
                                      gradient_param=gradient_parameters,
                                      d=d,
                                      data=data,
                                      mu=mu)
  logs = logistic_model.run_optimizer(n_epoch, log_freq, optimizer)

  # Visualize the logs
  file_tag = loss_type + "_mu_" + str(mu) + "_" + optimizer
  plot_loss(logs["iterations"],
            train_losses=logs["train_errors"],
            test_losses=logs["test_errors"],
            x_label="Epochs", y_label="Error",
            to_save=to_save, filename="error_" + file_tag)
  plot_loss(logs["iterations"],
            train_losses=logs["train_losses"],
            test_losses=logs["test_losses"],
            x_label="Epochs", y_label="Test loss",
            to_save=to_save, filename="loss_" + file_tag)

  # Visualize the predictions
  # predictions = list(logistic_model.predict(data["Xtrain"]))
  # predicted_labels = [int(round(prediction)) for prediction in predictions]
  # plot_data(data["Xtrain"], predicted_labels)


param_grid = {'optimizer': ["sag", "sgd"],
              'loss_type': ["square", "logistic"],
              'mu': [1e-1, 1e-5]
              }
# param_grid = {'optimizer': ["sgd"],
#               'loss_type': ["logistic"],
#               'mu': [1e-5]
#               }

grid = ParameterGrid(param_grid)
for params in grid:
  run(optimizer=params["optimizer"],
      loss_type=params["loss_type"],
      mu=params["mu"])
