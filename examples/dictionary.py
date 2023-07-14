#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:48:46 2020

@author: gab
"""

from abc import ABC, abstractmethod
import numpy as np

class Dictionary(ABC):
    @abstractmethod    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def sample(self):
        pass

class DeterministicDictionary(Dictionary):
    def __init__(self, X):
        super().__init__()
        self.X = X
        self.d = self.X.shape[0]
        
    def sample(self):
        return self.X

    
class RandomDictionary(Dictionary):
    def __init__(self, n, d):
        super().__init__()
        self.n = n
        self.d = d
    

class RandomDictionarySquare(RandomDictionary):
    def sample(self):
        return np.random.rand(self.n, self.d)


class RandomDictionaryDisk(RandomDictionary):
    def __init__(self, n):
        super().__init__(n, 2)
        
    def sample(self):
        r = np.random.rand(self.n, 1)
        theta = 2 * np.pi * np.random.rand(self.n)
        return np.sqrt(r) * np.c_[np.cos(theta), np.sin(theta)]


class RandomDictionarySphere(RandomDictionary):
    def __init__(self, n):
        super().__init__(n, 3)
        
    def sample(self):
        u = np.random.rand(self.n, 1)
        v = np.random.rand(self.n, 1)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi),  np.cos(phi)]


class RandomDictionaryThorus(RandomDictionary):
    def __init__(self, n):
        super().__init__(n, 3)
        
    def sample(self):
        """ https://math.stackexchange.com/questions/2017079/uniform-random-points-on-a-torus """
        R = 2
        r = 1
        points = np.empty((0, self.d))
        while points.shape[0] < self.n:
            n_left = self.n - points.shape[0]
            u = np.random.rand(n_left, 1)
            v = np.random.rand(n_left, 1)
            w = np.random.rand(n_left, 1)
            theta = 2 * np.pi * u
            phi = 2 * np.pi * v
            idx = np.nonzero(w <= (R + r * np.cos(theta)) / (R + r))
            new_points = np.c_[(R + r * np.cos(theta[idx])) * np.cos(phi[idx]), (R + r * np.cos(theta[idx])) * np.sin(phi[idx]), r * np.sin(theta[idx])]
            points = np.r_[points, new_points]

        return points



def main():
    import numpy as np
    import matplotlib.pyplot as plt

    d = 3
    n_random = 1000

    dictionary = RandomDictionaryThorus(n_random)

    X = dictionary.sample()
    
    if d == 2:
        fig = plt.figure(2)
        fig.clf()
        ax = fig.gca()
        ax.plot(X[:, 0], X[:, 1], '.')
        ax.grid()
        ax.axis('equal')
    elif d == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(2)
        fig.clf()
        ax = fig.gca(projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2], '.')
        ax.grid()
        ax.axis('equal')


if __name__ == '__main__':
    main()    
    
    











 