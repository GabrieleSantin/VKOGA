# Example to show the use of validation set tracking

import numpy as np
from matplotlib import pyplot as plt
from vkoga.kernels import Gaussian, Matern
from vkoga.vkoga import VKOGA

np.random.seed(1)


# Create some data (simply learn the identity)
dim = 2
X_train = np.random.rand(100, dim)
y_train = X_train

X_val = np.random.rand(1000, dim)
y_val = X_val


# Run VKOGA
kernel = Gaussian() # Matern(k=1)
model = VKOGA(kernel=kernel, greedy_type='f_greedy')
_ = model.fit(X_train, y_train, X_val=X_val, y_val=y_val, maxIter=50)


# Get ready for some plot
f_max, p_max = model.train_hist['f'], model.train_hist['p']
if model.flag_val:
    f_max_val, p_max_val = model.train_hist['f val'], model.train_hist['p val']

fig = plt.figure(2)
fig.clf()
plt.plot(f_max, 'r')
plt.plot(p_max, 'b')
plt.plot(model.train_hist['p selected'], 'k')
if model.flag_val:
    plt.plot(f_max_val, 'r--')
    plt.plot(p_max_val, 'b--')
plt.legend(['f max', 'p max'])
plt.xlabel('training iteration')
plt.yscale('log')
plt.draw()

if dim == 1:
    plt.figure(3)
    plt.clf()
    plt.plot(X_train, model.predict(X_train), '.')
    plt.plot(X_val, model.predict(X_val), '.')
    plt.draw()


# Result: Since the training set is quite small, we can clearly observe overfitting via the
# validation set tracking























