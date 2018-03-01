from scipy.io import loadmat

from logistic_regression import LogisticRegression
from tools import plot_loss
from tools import preprocess

# Parameters
N = 10000
n_iter = 5000
log_freq = 25
loss_type = "square"
gradient_stepsize = 1

# Import the data
mat = preprocess(data=loadmat('data_orsay_2017.mat'), N=N)

# Train
logistic_model = LogisticRegression(type=loss_type,
                                    gradient_stepsize=gradient_stepsize)

logistic_model.gradient_step(X=mat["Xtrain"],
                             y_true=mat["ytrain"])

logs = logistic_model.run_gradient_descent(mat, n_iter, log_freq)

file_tag = loss_type + "_" + str(N) + "_lr_" + str(gradient_stepsize)
plot_loss(logs["iterations"], logs["train_losses"], logs["test_losses"],
          y_label = "Loss", to_save=True, filename= "loss_" + file_tag)
plot_loss(logs["iterations"], logs["train_errors"], logs["test_errors"],
          y_label="Error", to_save=True, filename= "error_" + file_tag)