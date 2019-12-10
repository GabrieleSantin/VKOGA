#!/usr/bin/env python3

from kernels import Gaussian
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
    
# VKOGA implementation
class VKOGA(BaseEstimator):
                                          
    def __init__(self, kernel=Gaussian(), kernel_par=1,
                 verbose=True, 
                 greedy_type='p_greedy', reg_par=0, restr_par=0, 
                 tol_f=1e-10, tol_p=1e-10, max_iter=100):
        
        # Set the verbosity on/off
        self.verbose = verbose
        
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
            f_max = f[idx] 
            p_max = np.max(p)
        elif self.greedy_type == 'fp_greedy':
            idx = np.argmax(f[restr_idx][:, None] / p[restr_idx])
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
        
        if self.greedy_type == 'p_greedy':
            self.restr_par = 0
        
        indI = []
        notIndI = list(range(N))
        Vx = np.zeros((N, self.max_iter))
        if q > 1:
            c = np.zeros((self.max_iter, q))
        else:
            c = np.zeros(self.max_iter)
            
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
            # update the change of basis
            Cut_new_row = np.ones(n + 1)
            Cut_new_row[:n] = -Vx[indI[n], :n] @ Cut[:n:, :n]
            Cut[n, :n+1] = Cut_new_row / Vx[indI[n], n]      
            # compute the nth coefficient
            c[n] = y[indI[n]] / np.sqrt(p[indI[n]])
            # update the power function
            p[notIndI] = p[notIndI] - Vx[notIndI, n] ** 2
            # update the residual
            y[notIndI] = y[notIndI] - Vx[notIndI, n][:, None] * c[n]
            # remove the nth index from the dictionary
            notIndI.pop(idx)
        else:
            self.print_message('end')              

        # Define coefficients and centers
        c = c[:n+1]
        Cut = Cut[:n+1, :n+1]
        indI = indI[:n+1]
        self.coef_ = Cut.transpose() @ c
        self.ctrs_ = X[indI, :]
        f_max = np.sqrt(f_max[:n+1])
        p_max = np.sqrt(p_max[:n+1])

        return self


    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'coef_')

        # Validate the input
        X = check_array(X)
   
        # Evaluate the model
        return self.kernel.eval(X, self.ctrs_) @ self.coef_     


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
            



