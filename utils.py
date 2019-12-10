#!/usr/bin/env python3


import numpy as np
import scipy.stats as st

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
