import sys
import os
sys.path.append(os.getcwd() + "/src")

from logistic_regression import LogisticRegression
from datagen import DataGenerator
import matplotlib.pyplot as plt

from scipy.io import loadmat
from logistic_regression import LogisticRegression
from tools import plot_loss
from tools import preprocess, plot_data


# Parameters
N = 1000
n_iter = 1000
log_freq = 25
loss_type = "square"
d = 2

gradient_parameters = {"batch_size" : 50,
                       "stepsize_type" : "constant",
                       "initial_stepsize" : 1}


# Data generation

data_generator = DataGenerator(x_mu=[0,0],
                               x_sigma=[1,1],
                               x_dim=2,
                               noise_dev=0.5)
X, y = data_generator.generate(N)
data = data_generator.preprocess(X, y)

plot_data(X, y)

# Import the data
# data = preprocess(data=loadmat('../data_orsay_2017.mat'), N=N)

# Train
gradient_parameters["data"] = data
logistic_model = LogisticRegression(type=loss_type,
                                    gradient_param=gradient_parameters,
                                    d=d,
                                    data=data)
logs = logistic_model.run_gradient_descent(n_iter, log_freq)

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

predictions = list(logistic_model.predict(data["Xtrain"]))
predicted_labels = [int(round(prediction)) for prediction in predictions]

plot_data(data["Xtrain"], predicted_labels)
