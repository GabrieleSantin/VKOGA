#!/usr/bin/env python


#%% Import
import numpy as np
import matplotlib.pyplot as plt
from pgreedy import PGreedy






#%%
from pgreedy import RandomDictionary, DeterministicDictionary

d = 2
n_random = 1000
max_iter = 100

#X = np.random.rand(10000, d)
#dictionary = DeterministicDictionary(X)
dictionary = RandomDictionary(n_random, d)


#%%

#from kernels import Gaussian
#kernel = Gaussian(ep=1)

from kernels import Wendland
k = 3
kernel = Wendland(ep=.1, k=k, d=2)
exponent = (k + 2) / d - 1 / 2

#from kernels import Matern
#k = 3
#kernel = Matern(ep=1e-3, k=k)
#exponent = (2 * k ) / d - 1 / 2

#from kernels import Polynomial
#kernel = Polynomial(a=0, p=2)


#%%
model = PGreedy(kernel=kernel, max_iter=max_iter)
model.fit(dictionary)


#%%
p_max = []
np.random.seed(0)
Xte = np.random.rand(10000, d)

p_max = np.sqrt(model.predict_max(Xte, model.ctrs_.shape[0]))
 
    
#%%
n = model.ctrs_.shape[0]
tail = int(np.ceil(0.4 * n))
nn = np.arange(n - tail, n)

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
#fig.clf()
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
    ax.plot(model.ctrs_[:, 0], model.ctrs_[:, 1], 'o')
    ax.legend(['Selected points'])
    ax.grid()


##%%
#fig = plt.figure(3)
##fig.clf()
#ax = fig.gca()
##ax.plot(np.arange(4), E, 'o-')
#ax.plot(np.arange(4), np.polyval(cc, np.arange(4)))
#ax.grid()
