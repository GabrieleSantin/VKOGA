#!/usr/bin/env python3

from .kernels import Gaussian
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
    

# Pgreedy implementation
class PGreedy(BaseEstimator):
    """ VKOGA class restricted to P-greedy. It is used to generate only a 
    sequence of points and to compute the corresponding power function, without 
    fitting a function"""
    
    
    def __init__(self, kernel=Gaussian(), kernel_par=1,
                 verbose=True, n_report=100,
                 tol_p=1e-10, max_iter=100):
        
        # Set the verbosity on/off
        self.verbose = verbose
  
        # Set the frequency of report
        self.n_report = n_report
        
        # Set the params defining the method 
        self.kernel = kernel
        self.kernel_par = kernel_par
        
        # Set the stopping values
        self.max_iter = max_iter
        self.tol_p = tol_p
        
    def selection_rule(self, dictionary):
        # Sample a batch to compute the maximum
        X = dictionary.sample()
        # Compute the maximum
        p_X = self.predict(X)
        idx = np.argmax(p_X)

        return X[idx], p_X[idx]

    def fit(self, dictionary, reorth):
        # Initialize the convergence history (cold start)
        self.train_hist = {}
        self.train_hist['n'] = []
        self.train_hist['p'] = []
        
        # Initialize the kernel
        self.kernel.set_params(self.kernel_par)
        
        # Initialize the data-dependent variables
        self.ctrs_ = np.empty((0, dictionary.d))
        self.Cut_ = np.empty((0, 0))
         
        # Start
        self.print_message('begin')
        
        # Iterative selection of new points
        for n in range(self.max_iter):
           
            # Prepare the new history entry
            self.train_hist['n'].append(n+1)
            self.train_hist['p'].append([])
          
            # Select the current point
            x, self.train_hist['p'][n] = self.selection_rule(dictionary)
            
            # Check if the tolerances are reached
            if self.train_hist['p'][n] <= self.tol_p:
                n = n - 1
                self.print_message('end')   
                break
            
            # Evaluate the first (n-1) bases on the selected point
            if n > 0:
                Vx = self.kernel.eval(x, self.ctrs_) @ self.Cut_[:n, :n].transpose() 
            
            # Update the change of basis
            self.Cut_ = np.r_[np.c_[self.Cut_, np.zeros((n, 1))], np.zeros((1, n + 1))]
            new_row = np.ones((1, n + 1))
            if n > 0:
                new_row[0, :n] = (-Vx @ self.Cut_[:n, :n])
            self.Cut_[n, :] = new_row / np.sqrt(self.train_hist['p'][n])

#            if reorth == 1 and n > 0:
#                xx = np.sqrt(self.kernel.eval(x, x) - np.sum(Vx ** 2))
#                self.Cut_[n, -1] = 1 / xx
#                self.Cut_[n, :-1] = -(self.Cut_[:n, :n].transpose() @ Vx.transpose()).transpose() / xx
                
            
            # Add the current point to the selected centers
            self.ctrs_ = np.append(self.ctrs_, x[:, None].transpose(), axis=0)
            
            # Report some data every now and then
            if n % self.n_report == 0:
                self.print_message('track')              

        else:
            self.print_message('end')              

        return self


    def predict(self, X, n=None):
        # Try to do nothing
        if self.ctrs_ is None or n == 0:
            return self.kernel.diagonal(X)
        
        # Otherwise check if everything is ok
        # Check is fit has been called
        check_is_fitted(self, 'ctrs_')
        # Validate the input
        X = check_array(X)
   
        # Decide how many centers to use
        if n is None: 
            n = np.atleast_2d(self.ctrs_).shape[0]
        
        # Evaluate the power function on the input
        p = self.kernel.diagonal(X) - np.sum((self.kernel.eval(X, np.atleast_2d(self.ctrs_)[:n]) @ self.Cut_[:n, :n].transpose()) ** 2, axis=1)
        
        return p     
        


    def predict_max(self, X, n):
        # Check is fit has been called
        check_is_fitted(self, 'ctrs_')
        # Validate the input
        X = check_array(X)
   
        # Initialize        
        p_max = []
        p = self.kernel.diagonal(X)[:, None]
        p_max.append(np.max(p))
        A = self.kernel.eval(X, self.ctrs_)

        # Compute the maxima iteratively
        for i in range(n):
            p -= (A[:, :i+1] @ self.Cut_[i, :i+1][:, None]) ** 2
            p_max.append(np.max(p))   
            
        return p_max



    def print_message(self, when):
        
        if self.verbose and when == 'begin':
            print('')
            print('*' * 30 + ' [VKOGA] ' + '*' * 30)
            print('Training model with')
            print('       |_ kernel              : %s' % self.kernel)
            print('')
            
        if self.verbose and when == 'end':
            print('')
            print('Training completed with')
            print('       |_ selected points     : %8d / %8d' % (self.train_hist['n'][-1], self.max_iter))
            print('       |_ train power fun     : %2.2e / %2.2e' % (self.train_hist['p'][-1], self.tol_p))
            
        if self.verbose and when == 'track':
            print('Training ongoing with')
            print('       |_ selected points     : %8d / %8d' % (self.train_hist['n'][-1], self.max_iter))
            print('       |_ train power fun     : %2.2e / %2.2e' % (self.train_hist['p'][-1], self.tol_p))



