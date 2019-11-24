#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Oct 26 14:44:27 2019

@author: gab
'''
# Guardare se con PyTorch va meglio
#Check scikit-learn compatibility: https://scikit-learn.org/dev/developers/develop.html

from kernels import Kernel, RBF
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
    
class vkoga(BaseEstimator):
    ### TODO: Next steps  
    ### TODO:  * reorthogonalization
    ### TODO:  * decent comments & documentation
    ### TODO:  * jupyter
    ### TODO:  * test scikit compatibility
    ### TODO:  * guardare check_X_y, check_is_fitted(), ...
    
    def __init__(self, kernel=RBF(), verbose=True, 
                 greedy_type='p_greedy', reg_par=0, restr_par=0, 
                 tol_f=1e-10, tol_p=1e-10, max_iter=100):
        
        super(vkoga, self).__init__()
        
        # Set the verbosity on/off
        self.verbose = verbose
        
        # Set the params defining the method 
        self.kernel = kernel
        self.greedy_type = greedy_type
        self.reg_par = reg_par
        self.restr_par = restr_par
        
        # Set the stopping values
        self.max_iter = max_iter
        self.tol_f = tol_f
        self.tol_p = tol_p
        
    def selection_rule(self, f, p):
        # Move to 2 norm (for vectorial outputs) 
        f = np.sum(f ** 2, axis=1)
        if self.greedy_type == 'f_greedy':
            idx = np.argmax(f)
            f_max = f[idx] 
            p_max = np.max(p)
        elif self.greedy_type == 'fp_greedy':
            idx = np.argmax(f[:, None] / p)
            f_max = np.max(f)
            p_max = np.max(p)
        elif self.greedy_type == 'p_greedy':
            f_max = np.max(f)
            idx = np.argmax(p)
            p_max = p[idx]
        return idx, f_max, p_max
    
    def fit_(self, X, y):
        self.ctrs_ = X
        A = self.kernel.eval(X, X)
        self.coef_ = np.linalg.solve(A, y)

    def fit(self, X, y):
       
        # Initialize the model (cold start)
        self.coef_ = None
        self.ctrs_ = None
        
        # Initialize the convergence train_hist (cold start)
        self.train_hist = {}
        self.train_hist['n'] = []
        self.train_hist['f'] = []
        self.train_hist['p'] = []
        
        # Initialize the residual
        y = np.array(y)
        
        # Get the data dimension        
        N, q = y.shape         

        # Check the data dimension
        if X.shape[0] != N:
            raise ValueError('X.shape[0] should be the same as y.shape[0]')





        self.max_iter = min(self.max_iter, N)#Check
        
        
        indI = []
        notIndI = list(range(N))
        Vx = np.zeros((N, self.max_iter))
        c = np.zeros((self.max_iter, q))

        f_max = np.zeros((self.max_iter, 1))
        p_max = np.zeros((self.max_iter, 1))
        p = self.kernel.diagonal(X) + self.reg_par
        Cut = np.zeros((self.max_iter, self.max_iter))       
        
        self.print_message('begin')
        # Iterative selection of new points
        for n in range(self.max_iter):
            # prepare
            self.train_hist['n'].append(n+1)
            self.train_hist['f'].append([])
            self.train_hist['p'].append([])
            # select the current index
            idx, self.train_hist['f'][n], self.train_hist['p'][n] = self.selection_rule(y[notIndI], p[notIndI])
            # add the current index
            indI.append(notIndI[idx])
            # check if the tolerances are reacheded
            if self.train_hist['f'][n] <= self.tol_f:
                n = n - 1
                self.print_message('end')   
                break
            if self.train_hist['p'][n] <= self.tol_p:
                n = n - 1
                self.print_message('end')   
                break
            # compute the nth basis
            Vx[notIndI, n] = self.kernel.eval(X[notIndI, :], X[indI[n],:])[:, 0] - Vx[notIndI, :n+1] @ Vx[indI[n], 0:n+1].transpose()
            Vx[indI[n], n] += self.reg_par
            # normalize the nth basis
            Vx[notIndI, n] = Vx[notIndI, n] / np.sqrt(p[indI[n]])
#           reorthogonalization 
#           if n > 2:
#                breakpoint()
            # update the change of basis
            Cut_new_row = np.ones(n + 1)
            Cut_new_row[:n] = -Vx[indI[n], :n] @ Cut[:n:, :n]
            Cut[n, :n+1] = Cut_new_row / Vx[indI[n], n]      
            # compute the nth coefficient
            c[n, :] = y[indI[n], :] / np.sqrt(p[indI[n]])
            # update the power function
            p[notIndI] = p[notIndI] - np.atleast_2d(Vx[notIndI, n] ** 2).transpose()
            # update the residual
            y[notIndI, :] = y[notIndI, :] - np.atleast_2d(Vx[notIndI, n]).transpose() * np.atleast_2d(c[n, :])
            # remove the nth index from the dictionary
            notIndI.pop(idx)
        else:
            self.print_message('end')              

        # define coefficients and centers
        c = c[:n+1, :]
        Cut = Cut[:n+1, :n+1]
        indI = indI[:n+1]
        self.coef_ = Cut.transpose() @ c
        self.ctrs_ = X[indI, :]
        f_max = np.sqrt(f_max[:n+1]);
        p_max = np.sqrt(p_max[:n+1]);
        return self, self.train_hist['f'], self.train_hist['p']

    def predict(self, x):
        return self.kernel.eval(x, self.ctrs_) @ self.coef_     

    def print_message(self, when):
        
        if self.verbose and when == 'begin':
            print('*' * 30 + ' [VKOGA] ' + '*' * 30)
            print('Training model with')
            print('       |_ kernel              : %s' % self.kernel)
            print('       |_ regularization par. : %2.2e' % self.reg_par)
            print('       |_ restriction par.    : %2.2e' % self.restr_par)
            print('')
            
        if self.verbose and when == 'end':
            print('Training completed with')
            print('       |_ selected points     : %8d / %8d' % (self.train_hist['n'][-1], self.max_iter))
            print('       |_ train residual      : %2.2e / %2.2e' % (self.train_hist['f'][-1], self.tol_f))
            print('       |_ train power fun     : %2.2e / %2.2e' % (self.train_hist['p'][-1], self.tol_p))
            



#%%
from kernels import polynomial
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

params = {'reg_par': np.logspace(-10, 1, 10)}

#kernel=RBF(), verbose=True, greedy_type='p_greedy', reg_par=0, restr_par=0, tol_f=1e-10, tol_p=1e-10, max_iter=100)

model = GridSearchCV(vkoga(), params, n_jobs=1, cv=5, iid='False', refit=True, scoring='max_error')  


model.fit(X, y)







    