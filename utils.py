#!/usr/bin/env python3


import numpy as np
import scipy.stats as st
from scipy.spatial import distance_matrix

class log_uniform():        
# Solution from 
# https://stackoverflow.com/questions/49538120/how-to-implement-a-log-uniform-distribution-in-scipy
    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=None, random_state=None):
        uniform = st.uniform(loc=self.loc, scale=self.scale)
        if size is None:
            return np.power(self.base, uniform.rvs(random_state=random_state))
        else:
            return np.power(self.base, uniform.rvs(size=size, random_state=random_state))



def incremental_distance_matrix(X, batch_size):
    N, d = np.atleast_2d(X).shape
    num_batches = int(np.ceil(d / batch_size))
    D = np.zeros((N, N))
    for idx in range(num_batches):
        idx_begin = idx * batch_size
        idx_end = (idx + 1) * batch_size
        x = np.atleast_2d(X)[:, idx_begin:idx_end]
        D += distance_matrix(x, x) ** 2
        print('Added dimensions from %5d to %5d' %(idx_begin+1, np.min([d, idx_end])))
    return np.sqrt(D)