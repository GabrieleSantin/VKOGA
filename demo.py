#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:25:55 2019

@author: gab
"""

#%%
import numpy as np
from kernels import polynomial, RBF
from vkoga import vkoga
import matplotlib.pyplot as plt


#%%
ker = RBF(rbf_type='gauss', ep=4)
#kernel = RBF(rbf_type='mat2', ep=4)
#ker = RBF(rbf_type='wen', ep=4)
#ker = polynomial(a=0, p=2)

f = lambda x: np.array([np.cos(10 * x), np.sin(10 * x)])[:,:,0].transpose()
X = np.random.rand(10000, 2)
y = X

model = vkoga(kernel=ker)

_, f_max, p_max = model.fit(X, y)

X_te = np.random.rand(10000, 2)
s_te = model.predict(X_te)
y_te = X_te
s = model.predict(X)


#%%
fig = plt.figure(1)
fig.clf()
ax = fig.gca()
#ax.plot(X[:, 0], X[:, 1], '.')
ax.plot(model.ctrs_[:, 0], model.ctrs_[:, 1], '.')
ax.legend(['All points', 'Selected points'])
ax.grid()
fig.show()

fig = plt.figure(2)
fig.clf()
ax = fig.gca()
ax.loglog(f_max)
ax.loglog(p_max)
ax.legend(['f_max', 'p_max'])
ax.grid()
fig.show()




    
#from kernels import RBF, polynomial
#ker = RBF(rbf_type='gauss', ep=4)
##ker = polynomial(a=1, p=5)
#
#f = lambda x: np.array([np.cos(10 * x), np.sin(10 * x)])[:,:,0].transpose()
#X = np.linspace(-1, 1, 10000)[:, None]
#y = f(X)
#
#model = vkoga(ker)
#
#_, f_max, p_max = model.fit(X, y)
#
#X_te = np.linspace(-1, 1, 1000)[:, None]
#s_te = model.predict(X_te)
#y_te = f(X_te)
#s = model.predict(X)
#
##y_predicted = vkoga(C=100).fit(X_train, y_train).predict(X_test)
#
##%%
#fig = plt.figure(1)
#fig.clf()
#ax = fig.add_subplot(2, 1, 1)
#ax.plot(X, y[:, 0], 'o')
#ax.plot(X, s[:, 0], '.')
#ax.plot(X_te, y_te[:, 0], '-')
#ax.plot(X_te, s_te[:, 0], '-')
#ax.legend(['Train', 'Train prediction', 'Test', 'Test prediction'])
#ax.grid()
#
#ax = fig.add_subplot(2, 1, 2)
#ax.plot(X, y[:, 1], 'o')
#ax.plot(X, s[:, 1], '.')
#ax.plot(X_te, y_te[:, 1], '-')
#ax.plot(X_te, s_te[:, 1], '-')
#ax.legend(['Train', 'Train prediction', 'Test', 'Test prediction'])
#ax.grid()
##fig.show()
#
#fig = plt.figure(2)
#fig.clf()
#ax = fig.gca()
#ax.semilogy(f_max)
#ax.semilogy(p_max)
#ax.legend(['f_max', 'p_max'])
#ax.grid()
##fig.show()


#%%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def vectorial_max_error(y_true, y_pred):
    return np.max(np.sum((y_true - y_pred) ** 2, axis=1))

vectorial_score = make_scorer(vectorial_max_error, greater_is_better=False)

params = {'reg_par': np.logspace(-10, 1, 12)}

#kernel=RBF(), verbose=True, greedy_type='p_greedy', reg_par=0, restr_par=0, tol_f=1e-10, tol_p=1e-10, max_iter=100)
  

model = GridSearchCV(vkoga(verbose=False), params, n_jobs=1, cv=5, refit=True, verbose=2, scoring=vectorial_score)  


model.fit(X, y)

model.predict(X)


import pandas as pd
pd.DataFrame(model.cv_results_)




    