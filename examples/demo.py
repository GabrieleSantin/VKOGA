#!/usr/bin/env python

#%%
## Introduction to the basic usage of a VKOGA model

#%%
import numpy as np
import matplotlib.pyplot as plt


#%%
# Define a dataset `(X, y)` to run some experiments. In this case the map from the inputs to the outputs is just the identity.
X = np.random.rand(10000, 2)
y = X


#%%
# We split the dataset into a training (90% of the dataset) and a test set (10% of the dataset).
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)


#%%
### Basic training and prediction 

#%%
# We start by defining a VKOGA model with default parameters.
from vkoga.vkoga import VKOGA
model = VKOGA()


#%%
# By default, VKOGA uses a Gaussian kernel with shape parameter ep = 1. 
# The module `kernels` implements an abstact class `Kernel` and the concrete implementation of several kernels.
# For example, we can redefine the model to use a Gaussian kernel with ep = 4.
from vkoga.kernels import Gaussian
kernel = Gaussian(ep=4)
#from vkoga.kernels import Wendland
#kernel = Wendland(ep=2, k=0, d=2)
#from vkoga.kernels import Polynomial
#kernel = Polynomial(a=0, p=2)

model = VKOGA(kernel=kernel)


#%%
# The VKOGA model can now be trained on the dataset using the `fit` method:
_ = model.fit(X_train, y_train)


#%%
# The `fit` method prints some info (if `verbose = True`) and it returns the model itself (the omitted first output variable).
# After training, the information on the training history are stored in `train_hist`.
f_max, p_max = model.train_hist['f'], model.train_hist['p']

fig = plt.figure(1)
fig.clf()
ax = fig.gca()
ax.loglog(f_max)
ax.loglog(p_max)
ax.set_xlabel('Training iteration')
ax.legend(['Max training error', 'Max value of the power function'])
ax.grid()


#%%
# After the model is trained, the object `model` stores the coefficients `coef_` and the centers `ctrs_` of the kernel model.
fig = plt.figure(2)
fig.clf()
ax = fig.gca()
ax.plot(X[:, 0], X[:, 1], '.')
ax.plot(model.ctrs_[:, 0], model.ctrs_[:, 1], 'o')
ax.legend(['Training points', 'Selected points'])
ax.grid()


#%%
# In this case the model was trained with a fixed set of parameters. 
# The value of all the parameters can be obtained using the `get_params()` method.
model.get_params()


#%%
# These parameters can be set by the constructor (like we did with `kernel` above) or they can be modified with the `set_params()` method.
model.set_params(tol_f=1e-15)


#%%
# Once a model is trained, it can be used to compute predictions on new input data.
s_train = model.predict(X_train)
s_test = model.predict(X_test)


#%%
# And we can compute some errors.
err_train = np.max(np.linalg.norm(s_train - y_train, axis=1))
err_test = np.max(np.linalg.norm(s_test - y_test, axis=1))
print('Training error: %2.2e' % err_train)
print('Test error    : %2.2e' % err_test)


#%%
# Quick usage


#%%
# All these operations can also be condensed in a single line.
s_test = VKOGA(kernel=kernel).fit(X_train, y_train).predict(X_test)


#%%
# Refined training with parameter optimization


#%%
# The VKOGA models are compatible with scikit-learn interfaces, and in particular their parameters can be optimized with scikit-learn tools.
# First, one needs to define a score function to rank the models. The following is a simple vectorial version of the `max_error` scorer.
from sklearn.metrics import make_scorer

def vectorial_max_error(y_true, y_pred):
    return np.max(np.sum((y_true - y_pred) ** 2, axis=1))

vectorial_score = make_scorer(vectorial_max_error, greater_is_better=False)


#%% Deterministic parameter optimization


#%%
# For a deterministic parameter optimization, we first define a parameter search set.
params = {
        'reg_par': np.logspace(-16, 0, 5),
        'kernel_par': np.logspace(-1, 1, 5)
        }
  

#%%
# Then, we define the VKOGA model as a `GridSearchCV` object. In this case we run a 5-fold cross validation over the parameter set, using all the available cores (`n_jobs=-1`), and refitting the model on the entire training set after the validation is concluded. 
from sklearn.model_selection import GridSearchCV

model = GridSearchCV(VKOGA(verbose=False), params, scoring=vectorial_score, 
                     n_jobs=-1, cv=5, refit=True, verbose=1)  


#%%
# The model can be trained as before, but now in addition a deterministic cross validation is run to optimize the specified parameters.
model.fit(X, y)


#%%
# The parameters selected by the optimization are accessible as `best_params_`.
model.best_params_


#%%
# The detailed results of the parameter optimization process are instead in `cv_results_`.
import pandas as pd
pd.DataFrame(model.cv_results_)


#%%
# The trained model can be used as before to compute predictions.
s_train = model.predict(X_train)
s_test = model.predict(X_test)
err_train = np.max(np.linalg.norm(s_train - y_train, axis=1))
err_test = np.max(np.linalg.norm(s_test - y_test, axis=1))
print('Training error: %2.2e' % err_train)
print('Test error    : %2.2e' % err_test)


#%% Randomized parameter optimization


#%%
# In this case the parameters are randomly sampled according to some distribution, instead than on a grid.
from vkoga.utils import log_uniform
        
params = {'reg_par': log_uniform(-16, 1), 
         'kernel_par': log_uniform(-1, 1)
         }


#%%
# The VKOGA model is now defined as a `RandomizedSearchCV` object. In addition to the deterministic case, we need also to specify the number of samples to take from the parameter space (`n_iter=25`).
from sklearn.model_selection import RandomizedSearchCV

model = RandomizedSearchCV(VKOGA(verbose=False), params, scoring=vectorial_score, n_iter = 25, 
                           n_jobs=-1, cv=5, refit=True, verbose=1)


#%%
# Same training (with parameter optimization), parameter inspection and prediction as in the deterministic case
model.fit(X, y)

model.best_params_

pd.DataFrame(model.cv_results_).head()

s_train = model.predict(X_train)
s_test = model.predict(X_test)
err_train = np.max(np.linalg.norm(s_train - y_train, axis=1))
err_test = np.max(np.linalg.norm(s_test - y_test, axis=1))
print('Training error: %2.2e' % err_train)
print('Test error    : %2.2e' % err_test)


#%% Data preparation


#%%
# Scikit-learn provides also tools to preprocess the data.
# For example it is possible to define a scaler to normalize the data.
from sklearn import preprocessing
input_scaler = preprocessing.StandardScaler().fit(X_train)


#%%
# Or to scale them into a specific interval.
input_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X_train)


#%%
# Then, one can use the same scaler to compute predictions.
s_test = VKOGA().fit(input_scaler.transform(X_train), y_train).predict(input_scaler.transform(X_test))


#%%
# The same can be done also on the output data.

