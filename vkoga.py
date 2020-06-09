#!/usr/bin/env python3

from kernels import Gaussian
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
    
# VKOGA implementation
class VKOGA(BaseEstimator):
                                          
    def __init__(self, kernel=Gaussian(), kernel_par=1,
                 verbose=True, n_report=10,
                 greedy_type='p_greedy', reg_par=0, restr_par=0, 
                 tol_f=1e-10, tol_p=1e-10, max_iter=10):
        
        # Set the verbosity on/off
        self.verbose = verbose
        
        # Set the frequency of report
        self.n_report = n_report
        
        # Set the params defining the method 
        self.kernel = kernel
        self.kernel_par = kernel_par
        self.greedy_type = greedy_type
        self.reg_par = reg_par
        self.restr_par = restr_par
        
        # Set the stopping values
        self.max_iter = max_iter
        self.tol_f = tol_f
        self.tol_p = tol_p
        
    def selection_rule(self, f, p):
        if self.restr_par > 0:
            p_ = np.max(p)
            restr_idx = np.nonzero(p >= self.restr_par * p_)[0]
        else:
            restr_idx = np.arange(len(p))

        f = np.sum(f ** 2, axis=1)
        if self.greedy_type == 'f_greedy':
            idx = np.argmax(f[restr_idx])
            idx = restr_idx[idx]
            f_max = np.max(f)
            p_max = np.max(p)
        elif self.greedy_type == 'fp_greedy':
            idx = np.argmax(f[restr_idx] / p[restr_idx])
            idx = restr_idx[idx]
            f_max = np.max(f)
            p_max = np.max(p)
        elif self.greedy_type == 'p_greedy':
            f_max = np.max(f)
            idx = np.argmax(p)
            p_max = p[idx]
        return idx, f_max, p_max

    def fit(self, X, y):
        # Check the dataset
        X, y = check_X_y(X, y, multi_output=True)
        
        # Initialize the convergence history (cold start)
        self.train_hist = {}
        self.train_hist['n'] = []
        self.train_hist['f'] = []
        self.train_hist['p'] = []
        
        # Initialize the residual
        y = np.array(y)
        if len(y.shape) == 1:
            y = y[:, None]
        
        # Get the data dimension        
        N, q = y.shape

        self.max_iter = min(self.max_iter, N) 
        
        self.kernel.set_params(self.kernel_par)
        
        # Check compatibility of restriction
        if self.greedy_type == 'p_greedy':
            self.restr_par = 0
        if not self.reg_par == 0:
            self.restr_par = 0
        
        self.indI_  = []
        notIndI = list(range(N))
        Vx = np.zeros((N, self.max_iter))
        if q > 1:
            c = np.zeros((self.max_iter, q))
        else:
            c = np.zeros(self.max_iter)
            
        p = self.kernel.diagonal(X) + self.reg_par
        self.Cut_ = np.zeros((self.max_iter, self.max_iter))       
        
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
            self.indI_ .append(notIndI[idx])
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
            Vx[notIndI, n] = self.kernel.eval(X[notIndI, :], X[self.indI_ [n],:])[:, 0] - Vx[notIndI, :n+1] @ Vx[self.indI_ [n], 0:n+1].transpose()
            Vx[self.indI_ [n], n] += self.reg_par
            # normalize the nth basis
            Vx[notIndI, n] = Vx[notIndI, n] / np.sqrt(p[self.indI_ [n]])
            # update the change of basis
            Cut_new_row = np.ones(n + 1)
            Cut_new_row[:n] = -Vx[self.indI_ [n], :n] @ self.Cut_[:n:, :n]
            self.Cut_[n, :n+1] = Cut_new_row / Vx[self.indI_ [n], n]      
            # compute the nth coefficient
            c[n] = y[self.indI_ [n]] / np.sqrt(p[self.indI_ [n]])
            # update the power function
            p[notIndI] = p[notIndI] - Vx[notIndI, n] ** 2
            # update the residual
            y[notIndI] = y[notIndI] - Vx[notIndI, n][:, None] * c[n]
            # remove the nth index from the dictionary
            notIndI.pop(idx)
            
            # Report some data every now and then
            if n % self.n_report == 0:
                self.print_message('track')              

        else:
            self.print_message('end')              

        # Define coefficients and centers
        c = c[:n+1]
        self.Cut_ = self.Cut_[:n+1, :n+1]
        self.indI_  = self.indI_ [:n+1]
        self.coef_ = self.Cut_.transpose() @ c
        self.ctrs_ = X[self.indI_ , :]

        return self


    def predict(self, X):
        # Check is fit has been called
        check_is_fitted(self, 'coef_')

        # Validate the input
        X = check_array(X)
   
        # Evaluate the model
        return self.kernel.eval(X, self.ctrs_) @ self.coef_     
        ### TODO: replace with eval prod

    def print_message(self, when):
        
        if self.verbose and when == 'begin':
            print('')
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
                        
        if self.verbose and when == 'track':
            print('Training ongoing with')
            print('       |_ selected points     : %8d / %8d' % (self.train_hist['n'][-1], self.max_iter))
            print('       |_ train residual      : %2.2e / %2.2e' % (self.train_hist['f'][-1], self.tol_f))
            print('       |_ train power fun     : %2.2e / %2.2e' % (self.train_hist['p'][-1], self.tol_p))


#%% Utilities to 
import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):        
    with open(filename, 'rb') as input:
        obj = pickle.load(input)    
    return obj



