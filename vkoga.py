#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Oct 26 14:44:27 2019

@author: gab
'''
# Guardare se con PyTorch va meglio
#Check scikit-learn compatibility: https://scikit-learn.org/dev/developers/develop.html

from kernels import Kernel
import numpy as np
import matplotlib.pyplot as plt

    
class vkoga(object):
    ### TODO: Next steps  
    ### TODO:  * parameter parsing (like scikit) 
    ### TODO:  * reorth.
    ### TODO:  * decent comments & documentation
    ### TODO:  * stampa messaggio (con str(kernel))
    ### TODO:  * rimuovere Afun
    ### TODO:  * jupyter
    ### TODO:  * test scikit compatibility

    
    
    def __init__(self, kernel):
        self.kernel = kernel
        self.coef_ = None
        self.ctrs_ = None
        self.verbose = True
     
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
        y = np.array(y)
        self.greedy_type = 'p_greedy'
        reg_par = 1e-5
        tol_F = 1e-10
        tol_P = 1e-12
        max_iter = 100
        
        N, q = y.shape         
        max_iter = min(max_iter, N)
        indI = []
        notIndI = list(range(N))
        Vx = np.zeros((N, max_iter))
        c = np.zeros((max_iter, q))
        f_max = np.zeros((max_iter, 1))
        p_max = np.zeros((max_iter, 1))
        p = self.kernel.diagonal(X) + reg_par
        Cut = np.zeros((max_iter, max_iter))       
        
        # Thus was useful in the Matlab version to avoid passing X, y to this 
        # function. But here it is required by the scikit-learn interface
        Afun = lambda i, j: ker.eval(X[i, :], X[j, :]) #+ reg_par * (i[:, None] == j[:, None].transpose())
        
        # Iterative selection of new points
        for n in range(max_iter):
            # select the current index
            idx, f_max[n], p_max[n] = self.selection_rule(y[notIndI], p[notIndI])
            # add the current index
            indI.append(notIndI[idx])
            # check if the tolerances are reacheded
            if f_max[n] <= tol_F:
                n = n - 1
                self.print_message('f-tolerance reached: stop')   
                break
            if p_max[n] <= tol_P:
                n = n - 1
                self.print_message('p-tolerance reached: stop')   
                break
            # If not, add the current point    
            # compute the nth basis
            Vx[notIndI, n] = Afun(notIndI, indI[n])[:, 0] - Vx[notIndI, :n+1] @ Vx[indI[n], 0:n+1].transpose()
            # normalize the nth basis
            Vx[notIndI, n] = Vx[notIndI, n] / np.sqrt(p[indI[n]])
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
            self.print_message('max_iter reached: stop')              

        # define coefficients and centers
        c = c[:n+1, :]
        Cut = Cut[:n+1, :n+1]
        indI = indI[:n+1]
        self.coef_ = Cut.transpose() @ c
        self.ctrs_ = X[indI, :]
#        fmax = np.sqrt(fmax(1 : n));
#        pmax = np.sqrt(pmax(1 : n));
        return self, f_max, p_max

    def predict(self, x):
        return self.kernel.eval(x, self.ctrs_) @ self.coef_     

    def print_message(self, message):
        base_str = '[VKOGA] ' 
        base_str += 'Iteration terminated after n = %2.d' % self.coef_.shape[0]  
        if self.verbose:
            print(base_str + message)

#from kernels import RBF, polynomial
#ker = RBF(rbf_type='gauss', ep=4)
##ker = RBF(rbf_type='mat2', ep=4)
##ker = polynomial(a=0, p=2)
#
#f = lambda x: np.array([np.cos(10 * x), np.sin(10 * x)])[:,:,0].transpose()
#X = np.random.rand(1000, 2)
#y = X
#
#model = vkoga(ker)
#
#_, f_max, p_max = model.fit(X, y)
#
#X_te = np.random.rand(10000, 2)
#s_te = model.predict(X_te)
#y_te = X_te
#s = model.predict(X)
#
#
##%%
#fig = plt.figure(1)
#fig.clf()
#ax = fig.gca()
#ax.plot(X[:, 0], X[:, 1], '.')
#ax.plot(model.ctrs_[:, 0], model.ctrs_[:, 1], 'o')
#ax.legend(['All points', 'Selected points'])
#ax.grid()
#fig.show()
#
#fig = plt.figure(2)
#fig.clf()
#ax = fig.gca()
#ax.loglog(f_max)
#ax.loglog(p_max)
#ax.legend(['f_max', 'p_max'])
#ax.grid()
#fig.show()




    
from kernels import RBF, polynomial
ker = RBF(rbf_type='gauss', ep=4)
#ker = polynomial(a=0, p=5)

f = lambda x: np.array([np.cos(10 * x), np.sin(10 * x)])[:,:,0].transpose()
X = np.linspace(-1, 1, 10000)[:, None]
y = f(X)

model = vkoga(ker)

_, f_max, p_max = model.fit(X, y)

X_te = np.linspace(-1, 1, 1000)[:, None]
s_te = model.predict(X_te)
y_te = f(X_te)
s = model.predict(X)

#y_predicted = vkoga(C=100).fit(X_train, y_train).predict(X_test)

#%%
fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(2, 1, 1)
ax.plot(X, y[:, 0], 'o')
ax.plot(X, s[:, 0], '.')
ax.plot(X_te, y_te[:, 0], '-')
ax.plot(X_te, s_te[:, 0], '-')
ax.legend(['Train', 'Train prediction', 'Test', 'Test prediction'])
ax.grid()

ax = fig.add_subplot(2, 1, 2)
ax.plot(X, y[:, 1], 'o')
ax.plot(X, s[:, 1], '.')
ax.plot(X_te, y_te[:, 1], '-')
ax.plot(X_te, s_te[:, 1], '-')
ax.legend(['Train', 'Train prediction', 'Test', 'Test prediction'])
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




    