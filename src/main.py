import os
import sys
sys.path.append(os.getcwd() + "/src")

from datagen import DataGenerator
from logistic_regression import LogisticRegression
from tools import plot_loss, plot_data, preprocess
from scipy.io import loadmat


# Parameters
N = 1000
n_iter = 1000
log_freq = 25
loss_type = "logistic"
d = 2
synthetic_data = True

gradient_parameters = {"batch_size" : 50,
                       "stepsize_type" : "constant",
                       "initial_stepsize" : 1}


# Data generation
if synthetic_data :
  data_generator = DataGenerator(x_mu=[0] * d,
                                 x_sigma=[1] * d,
                                 x_dim=d,
                                 noise_dev=0.8)
  data = data_generator.preprocess(X, y)
else :
  data = preprocess(data=loadmat('../data_orsay_2017.mat'), N=N)


plot_data(data["Xtrain"], data["ytrain"])

# Train
gradient_parameters["data"] = data
logistic_model = LogisticRegression(type=loss_type,
                                    gradient_param=gradient_parameters,
                                    d=d,
                                    data=data)
logs = logistic_model.run_gradient_descent(n_iter, log_freq)

# Visualize the logs
file_tag = loss_type + "_" + str(N) + "_lr_" + \
           str(gradient_parameters["initial_stepsize"])
plot_loss(logs["iterations"],
          train_losses=logs["train_losses"],
          test_losses=logs["test_losses"],
          y_label = "Loss", to_save=False, filename= "loss_" + file_tag)
plot_loss(logs["iterations"],
          train_losses=logs["train_errors"],
          test_losses=logs["test_errors"],
          y_label="Error", to_save=False, filename= "error_" + file_tag)


# Visualize the predictions
predictions = list(logistic_model.predict(data["Xtrain"]))
predicted_labels = [int(round(prediction)) for prediction in predictions]
plot_data(data["Xtrain"], predicted_labels)
