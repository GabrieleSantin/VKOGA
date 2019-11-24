from abc import ABC, abstractmethod
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt

class Kernel(ABC):
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def diagonal(self, X):
        pass


class RBF(Kernel):
    def __init__(self, rbf_type='gauss', ep=1):
        self.rbf_type = rbf_type
        self.ep = ep
        if self.rbf_type == 'gauss':
            self.rbf = lambda ep, r: np.exp(-(ep * r) ** 2)
        elif self.rbf_type == 'gauss_tanh':
            self.rbf = lambda ep, r: np.exp(-(ep * np.tanh(r)) ** 2)
        elif self.rbf_type == 'imq':
            self.rbf = lambda ep, r: 1. / np.sqrt(1 + (ep * r) ** 2)
        elif self.rbf_type == 'mat0':
            self.rbf = lambda ep, r : np.exp(-ep * r)
        elif self.rbf_type == 'mat1':
            self.rbf = lambda ep, r: np.exp(-ep * r) * (1 + ep * r)
        elif self.rbf_type == 'mat2':
            self.rbf = lambda ep, r: np.exp(-ep * r) * (3 + 3 * ep * r + (ep * r) ** 2)
        elif self.rbf_type ==  'wen':   
            k = 0
            d = 2
            l = np.floor(d / 2) + k + 1
            c = np.math.factorial(l + 2 * k) / np.math.factorial(l)
            e = l + k
#            if k == 0:
#                p = '1';
#            elif k == 1:
#                p = '(l + 1) * r + 1'
#            elif k == 2:
#                p = '(l + 3) * (l + 1) * r .^ 2 + 3 * (l + 2) * r + 3'
#            elif k == 3:
#                p = '(l + 5) * (l + 3) * (l + 1) * r .^ 3 + (45 + 6 * l * (l + 6)) * r .^ 2 + (15 * (l + 3)) * r + 15'
#            elif k == 4:
#                p = '(l + 7) * (l + 5) * (l + 3) * (l + 1) * r .^ 4 + (5 * (l + 4) * (21 + 2 * l * (8 + l))) * r .^ 3 + (45 * (14 + l * (l + 8))) * r .^ 2 + (105 * (l + 4)) * r + 105'
#            elif:
#                raise Exception('This Wendland kernel is not implemented')
            self.rbf = lambda ep, r: np.max(1 - ep * r, 0) ** e * ((l + 1) * ep * r + 1) / c
           
        else:
            self.rbf = None
            raise Exception('This RBF is not implemented')
            
    def eval(self, x, y):
        return self.rbf(self.ep, distance_matrix(np.atleast_2d(x), np.atleast_2d(y)))

    def diagonal(self, X):
        return np.ones((X.shape[0], 1)) * self.rbf(self.ep, 0)
    
    def __str__(self):
     return self.rbf_type + ' [gamma = %2.2e]' % self.ep   

    
class polynomial(Kernel):
    def __init__(self, a=0, p=1):
        self.a = a
        self.p = p
            
    def eval(self, x, y):
        return (np.atleast_2d(x) @ np.atleast_2d(y).transpose() + self.a) ** self.p
    
    def diagonal(self, X):
        return ((np.linalg.norm(X, axis=1) + self.a) ** self.p)[:, None]

    def __str__(self):
     return 'polynomial' + ' [a = %2.2e, p = %2.2e]' % (self.a, self.p)   


def main():
    ker = RBF(rbf_type='gauss')

    x = np.linspace(-1, 1, 100)[:, None]
    y = np.matrix([0])
    A = ker.eval(x, y)


    fig = plt.figure(1)
    fig.clf()
    ax = fig.gca()
    ax.plot(A)
    ax.set_title('A kernel')
    fig.show()


if __name__ == '__main__':
    main()


        