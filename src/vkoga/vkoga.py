#!/usr/bin/env python3

from .kernels import Gaussian
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.spatial import distance_matrix
    
# VKOGA implementation
class VKOGA(BaseEstimator):
                                          
    def __init__(self, kernel=Gaussian(), kernel_par=1,
                 verbose=True, n_report=10,
                 greedy_type='p_greedy', reg_par=0, restr_par=0, 
                 tol_f=1e-10, tol_p=1e-10):
        
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

        self.flag_val = None
        
        # Set the stopping values
        self.tol_f = tol_f
        self.tol_p = tol_p

        # Some further settings
        self.ctrs_ = None
        self.Cut_ = None
        self.c = None


        # Initialize the convergence history
        self.train_hist = {}
        self.train_hist['n'] = []
        self.train_hist['f'] = []
        self.train_hist['p'] = []
        self.train_hist['p selected'] = []              # list of selected power vals
        self.train_hist['f val'] = []
        self.train_hist['p val'] = []
        
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

    def fit(self, X, y, X_val=None, y_val=None, maxIter=None):
        # Check the datasets
        X, y = check_X_y(X, y, multi_output=True)
        if len(y.shape) == 1:
            y = y[:, None]

        if X_val is None or y_val is None:
            X_val = None
            y_val = None
            self.flag_val = False
        else:
            self.flag_val = True
            X_val, y_val = check_X_y(X_val, y_val, multi_output=True)
            # We will assume in the following that X_val and X are disjoint

            if len(y_val.shape) == 1:
                y_val = y_val[:, None]

        # Check whether already fitted
        if self.ctrs_ is None:
            self.ctrs_ = np.zeros((0, X.shape[1]))
            self.Cut_ = np.zeros((0, 0))
            self.c = np.zeros((0, y.shape[1]))


        # Check whether "new X" and previously chosen centers overlap
        list_truly_new_X = []
        if self.ctrs_.shape[0] > 0:
            for idx_x in range(X.shape[0]):
                if min(np.linalg.norm(self.ctrs_ - X[idx_x, :], axis=1)) < 1e-10:
                    continue
                else:
                    list_truly_new_X.append(idx_x)
        else:
            list_truly_new_X = list(range(X.shape[0]))
        X = X[list_truly_new_X, :]
        y = y[list_truly_new_X, :]



        # Initialize the residual and update the given y values by substracting the current model
        y = np.array(y)
        if len(y.shape) == 1:
            y = y[:, None]
        y = y - self.predict(X)
        if self.flag_val:
            y_val = y_val - self.predict(X_val)


        # Get the data dimension
        N, q = y.shape
        if self.flag_val:
            N_val = y_val.shape[0]


        # Set maxIter_continue
        if maxIter is None or maxIter > N:
            self.maxIter = 100
        else:
            self.maxIter = maxIter


        # Check compatibility of restriction
        if self.greedy_type == 'p_greedy':
            self.restr_par = 0
        if not self.reg_par == 0:
            self.restr_par = 0


        # Initialize list for the chosen and non-chosen indices
        indI_ = []
        notIndI = list(range(N))
        c = np.zeros((self.maxIter, q))


        # Compute the Newton basis values (related to the old centers) on the new X
        Vx_new_X_old_ctrs = self.kernel.eval(X, self.ctrs_) @ self.Cut_.transpose()
        if self.flag_val:
            Vx_val_new_X_old_ctrs = self.kernel.eval(X_val, self.ctrs_) @ self.Cut_.transpose()


        # Initialize arrays for the Newton basis values (related to the new centers) on the new X
        Vx = np.zeros((N, self.maxIter))
        if self.flag_val:
            Vx_val = np.zeros((N_val, self.maxIter))


        # Compute the powervals on X and X_val
        p = self.kernel.diagonal(X) + self.reg_par
        p = p - np.sum(Vx_new_X_old_ctrs ** 2, axis=1)
        if self.flag_val:
            p_val = self.kernel.diagonal(X_val) + self.reg_par
            p_val = p_val - np.sum(Vx_val_new_X_old_ctrs ** 2, axis=1)


        # Extend Cut_ matrix, i.e. continue to build on old self.Cut_ matrix
        N_ctrs_so_far = self.Cut_.shape[0]
        Cut_ = np.zeros((N_ctrs_so_far + self.maxIter, N_ctrs_so_far + self.maxIter))
        Cut_[:N_ctrs_so_far, :N_ctrs_so_far] = self.Cut_


        # Iterative selection of new points
        self.print_message('begin')
        for n in range(self.maxIter):
            # prepare
            self.train_hist['n'].append(self.ctrs_.shape[0] + n + 1)
            self.train_hist['f'].append([])
            self.train_hist['p'].append([])
            self.train_hist['p selected'].append([])
            if self.flag_val:
                self.train_hist['p val'].append([])
                self.train_hist['f val'].append([])

            # select the current index
            idx, self.train_hist['f'][-1], self.train_hist['p'][-1] = self.selection_rule(y[notIndI], p[notIndI])
            self.train_hist['p selected'][-1] = p[notIndI[idx]]
            if self.flag_val:
                self.train_hist['p val'][-1] = np.max(p_val)
                self.train_hist['f val'][-1] = np.max(np.sum(y_val ** 2, axis=1))

            # add the current index
            indI_.append(notIndI[idx])

            # check if the tolerances are reacheded
            if self.train_hist['f'][n] <= self.tol_f:
                n = n - 1
                self.print_message('end')
                break
            if self.train_hist['p'][n] <= self.tol_p:
                n = n - 1
                self.print_message('end')
                break

            # compute the nth basis (including normalization)# ToDo: Also old Vx need to be substracted here!
            Vx[notIndI, n] = self.kernel.eval(X[notIndI, :], X[indI_[n], :])[:, 0]\
                 - Vx_new_X_old_ctrs[notIndI, :] @ Vx_new_X_old_ctrs[indI_[n], :].transpose()\
                 - Vx[notIndI, :n+1] @ Vx[indI_[n], 0:n+1].transpose()
            Vx[indI_[n], n] += self.reg_par
            Vx[notIndI, n] = Vx[notIndI, n] / np.sqrt(p[indI_[n]])

            if self.flag_val:
                Vx_val[:, n] = self.kernel.eval(X_val, X[indI_[n], :])[:, 0]\
                    - Vx_val_new_X_old_ctrs[:, :] @ Vx_new_X_old_ctrs[indI_[n], :].transpose()\
                    - Vx_val[:, :n+1] @ Vx[indI_[n], 0:n+1].transpose()
                Vx_val[:, n] = Vx_val[:, n] / np.sqrt(p[indI_[n]])


            # update the change of basis
            Cut_new_row = np.ones(N_ctrs_so_far + n + 1)
            Cut_new_row[:N_ctrs_so_far + n] = \
                -np.concatenate((Vx_new_X_old_ctrs[indI_[n], :], Vx[indI_[n], :n])) \
                @ Cut_[:N_ctrs_so_far + n, :N_ctrs_so_far + n]
            Cut_[N_ctrs_so_far + n, :N_ctrs_so_far + n + 1] = Cut_new_row / Vx[indI_[n], n]

            # compute the nth coefficient
            c[n] = y[indI_[n]] / np.sqrt(p[indI_[n]])

            # update the power function
            p[notIndI] = p[notIndI] - Vx[notIndI, n] ** 2
            if self.flag_val:
                p_val = p_val - Vx_val[:, n] ** 2

            # update the residual
            y[notIndI] = y[notIndI] - Vx[notIndI, n][:, None] * c[n]
            if self.flag_val:
                y_val = y_val - Vx_val[:, n][:, None] * c[n]

            # remove the nth index from the dictionary
            notIndI.pop(idx)

            # Report some data every now and then
            if n % self.n_report == 0:
                self.print_message('track')

        else:
            self.print_message('end')

        # Define coefficients and centers
        self.c =  np.concatenate((self.c, c[:n + 1]))
        self.Cut_ = Cut_[:N_ctrs_so_far + n + 1, :N_ctrs_so_far + n + 1]
        self.indI_ = indI_[:n + 1]     # Mind: These are only the indices of the latest points
        self.coef_ = self.Cut_.transpose() @ self.c
        self.ctrs_ = np.concatenate((self.ctrs_, X[self.indI_, :]), axis=0)


        return self


    def predict(self, X):
        # Check is fit has been called
        # check_is_fitted(self, 'coef_')     # ToDo: Remove this one!

        # Validate the input
        X = check_array(X)

        # Compute prediction
        if self.ctrs_.shape[0] > 0:
            prediction = self.kernel.eval(X, self.ctrs_) @ self.coef_
        else:
            prediction = np.zeros((X.shape[0], 1))

        return prediction
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
            print('       |_ selected points     : %8d / %8d' % (self.train_hist['n'][-1], self.ctrs_.shape[0] + self.maxIter))
            if self.flag_val:
                print('       |_ train, val residual : %2.2e / %2.2e,    %2.2e' %
                      (self.train_hist['f'][-1], self.tol_f, self.train_hist['f val'][-1]))
                print('       |_ train, val power fun: %2.2e / %2.2e,    %2.2e' %
                      (self.train_hist['p'][-1], self.tol_p, self.train_hist['p val'][-1]))
            else:
                print('       |_ train residual      : %2.2e / %2.2e' % (self.train_hist['f'][-1], self.tol_f))
                print('       |_ train power fun     : %2.2e / %2.2e' % (self.train_hist['p'][-1], self.tol_p))
                        
        if self.verbose and when == 'track':
            print('Training ongoing with')
            print('       |_ selected points     : %8d / %8d' % (self.train_hist['n'][-1], self.ctrs_.shape[0] + self.maxIter))
            if self.flag_val:
                print('       |_ train, val residual : %2.2e / %2.2e,    %2.2e' %
                      (self.train_hist['f'][-1], self.tol_f, self.train_hist['f val'][-1]))
                print('       |_ train, val power fun: %2.2e / %2.2e,    %2.2e' %
                      (self.train_hist['p'][-1], self.tol_p, self.train_hist['p val'][-1]))
            else:
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



