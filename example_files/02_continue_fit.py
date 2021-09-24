# Example to show the use of the continue_fit function

import numpy as np
from matplotlib import pyplot as plt
from vkoga.kernels import Gaussian, Matern
from vkoga.vkoga import VKOGA

np.random.seed(1)


# Create some data
dim = 2
X_train = np.random.rand(3000, dim)
y_train = X_train

X_val = np.random.rand(10000, dim)
y_val = X_val


# Choose a kernel
kernel = Gaussian() # Matern(k=1)


# First VKOGA: Run directly for 30 centers
model1 = VKOGA(kernel=kernel)
model1.fit(X_train, y_train, X_val=X_val, y_val=y_val, maxIter=30)

# Second VKOGA: Run three times, each for 10 centers
model2 = VKOGA(kernel=kernel)
model2.fit(X_train, y_train, X_val=X_val, y_val=y_val, maxIter=10)
model2.fit(X_train, y_train, X_val=X_val, y_val=y_val, maxIter=10)
model2.fit(X_train, y_train, X_val=X_val, y_val=y_val, maxIter=10)


# Compare model1 and model2
assert np.linalg.norm(np.array(model1.train_hist['f']) - np.array(model2.train_hist['f'])) < 1e-12,         'Deviation too large for f'
assert np.linalg.norm(np.array(model1.train_hist['f val']) - np.array(model2.train_hist['f val'])) < 1e-12, 'Deviation too large for f val'
assert np.linalg.norm(np.array(model1.train_hist['p']) - np.array(model2.train_hist['p'])) < 1e-12,         'Deviation too large for p'
assert np.linalg.norm(np.array(model1.train_hist['p val']) - np.array(model2.train_hist['p val'])) < 1e-12, 'Deviation too large for p val'



























