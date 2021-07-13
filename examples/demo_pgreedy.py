ori#!/usr/bin/env python


#%% Import
import numpy as np
import matplotlib.pyplot as plt
from vkoga.pgreedy import PGreedy






#%%
d = 2
n_random = 1000
max_iter = int(1e4)
tol_p = 1e-12

from dictionary import RandomDictionarySquare, RandomDictionaryDisk, DeterministicDictionary, RandomDictionarySphere, RandomDictionaryThorus

#X = np.random.rand(10000, d)
#dictionary = DeterministicDictionary(X)
dictionary = RandomDictionaryDisk(n_random)
#dictionary = RandomDictionaryThorus(n_random)

#dictionary_test = RandomDictionaryThorus(10000)
dictionary_test = RandomDictionaryDisk(10000)


#%%
#from vkoga.kernels import Gaussian
#kernel = Gaussian()
#ep = 1

from vkoga.kernels import Wendland
k = 3
ep = .1
kernel = Wendland(k=k, d=2)
#exponent = (k + 2) / d - 1 / 2

#from vkoga.kernels import Matern
#k = 3
#kernel = Matern(ep=1e-3, k=k)
#exponent = (2 * k ) / d - 1 / 2

#from vkoga.kernels import Polynomial
#kernel = Polynomial(a=0, p=2)


#%%
model = PGreedy(kernel=kernel, kernel_par=ep, max_iter=max_iter, tol_p=tol_p)
model.fit(dictionary, 0)


#%%
p_max = []
np.random.seed(0)

Xte = dictionary_test.sample()

p_max = np.sqrt(model.predict_max(Xte, model.ctrs_.shape[0]))
 
    
#%%
n = model.ctrs_.shape[0]
tail = int(np.ceil(0.3 * n))
nn = np.arange(n - tail, n )

if kernel.name == 'gauss':
#    c = np.polyfit(np.sqrt(nn), np.log(p_max[-tail:]), 1)
#    nn = np.arange(n)
#    nn_rate = np.exp(c[0] * np.sqrt(nn) + c[1])
#    rate_leg_string = 'exp(c sqrt(n)), c = %2.2f (estimated)' %c[0]
    c = np.polyfit(nn, np.log(p_max[-tail:]), 1)
    nn = np.arange(n)
    nn_rate = np.exp(c[0] * nn + c[1])
    rate_leg_string = 'exp(c n), c = %2.2f (estimated)' %c[0]
else:
    c = np.polyfit(nn, np.log(p_max[-tail:]) / np.log(nn), 0)   
    tau_data = -(c - 1/2) * d
    nn = np.arange(1, n)
    nn_rate = nn ** (-tau_data/d + 1/2)
    rate_leg_string = 'tau = %2.2f (estimated)' %tau_data

#E.append(tau_data)


#%%   
fig = plt.figure(1)
fig.clf()
ax = fig.gca()
if kernel.name == 'gauss':
    ax.semilogy(p_max, '.-')
    ax.semilogy(nn, nn_rate)
else:
    ax.loglog(p_max, '.-')
    ax.loglog(nn, nn_rate)
ax.set_xlabel('Training iteration')
ax.legend(['Max value of the power function', rate_leg_string])
ax.grid()


if d == 2:
    fig = plt.figure(2)
    fig.clf()
    ax = fig.gca()
    ax.plot(model.ctrs_[:, 0], model.ctrs_[:, 1], '.')
    ax.legend(['Selected points'])
    ax.grid()
    ax.axis('equal')
elif d == 3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(2)
    fig.clf()
    ax = fig.gca(projection='3d')
    ax.plot(model.ctrs_[:, 0], model.ctrs_[:, 1], model.ctrs_[:, 2], '.')
    ax.legend(['Selected points'])
    ax.grid()
#    ax.axis('equal')


#%%
#fig = plt.figure(3)
#fig.clf()
#ax = fig.gca()
#ax.legend(['Selected points'])
#ax.grid()
#for k in range(n):    
#    ax.plot(model.ctrs_[k, 0], model.ctrs_[k, 1], '.')
#    plt.pause(0.1)


##%%
#fig = plt.figure(3)
##fig.clf()
#ax = fig.gca()
##ax.plot(np.arange(4), E, 'o-')
#ax.plot(np.arange(4), np.polyval(cc, np.arange(4)))
#ax.grid()


#%% Orthogonality check
A = model.kernel.eval(model.ctrs_, model.ctrs_)

G = model.Cut_ @ A @ model.Cut_.transpose()    
    
fig = plt.figure(4)
fig.clf()
ax = fig.gca()
ax.grid()
ax.plot(G.transpose()) 
    
print('Max abs (1 - diagonal) = %2.2e' % np.max(np.abs(1 - np.diag(G))))    
print('Max abs off diagonal = %2.2e' % np.max(np.abs(G - np.diag(np.diag(G)))))    
print('Norm(G - eye(n)) = %2.2e' % np.linalg.norm(G - np.eye(n, n)))    
    
    
    
    
    
    
    
    
    
    
