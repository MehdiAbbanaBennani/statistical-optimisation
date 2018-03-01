from scipy.io import loadmat
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from tqdm import tqdm

from tools import plot_loss

# Parameters
n_iter = 100
log_freq = 10

# Import the data
mat = loadmat('data_orsay_2017.mat')

Xtrain = mat["Xtrain"]
ytrain = mat["ytrain"]
Xtest = mat["Xtest"]
ytest = mat["ytest"]

# wtest_hinge = mat["wtest_hinge"]
# wtest_square = mat["wtest_square"]
# wtest_logistic = mat["wtest_logistic"]

# Log arrays
train_losses = {"square": [],
                "log": []}
test_losses = {"square": [],
               "log": []}
iterations = []


# Train
clf = SGDClassifier(loss="log", penalty="none",
                    learning_rate="constant", eta0=0.01,
                    warm_start=True,
                    n_iter=1)

for it in tqdm(range(n_iter)):
    clf.fit(Xtrain, ytrain)

    if it % log_freq == 0 :
        ytrain_pred = clf.predict(Xtrain)
        ytest_pred = clf.predict(Xtest)

        iterations.append(it)
        train_losses["log"].append(log_loss(y_true=ytrain, y_pred=ytrain_pred))
        test_losses["log"].append(log_loss(y_true=ytest, y_pred=ytest_pred))

plot_loss(iterations, train_losses, test_losses)
