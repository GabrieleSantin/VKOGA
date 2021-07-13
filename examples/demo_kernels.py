import numpy as np
from vkoga.kernels import Gaussian
import matplotlib.pyplot as plt


ker = Gaussian()

x = np.linspace(-1, 1, 100)[:, None]
y = np.matrix([0])
A = ker.eval(x, y)


fig = plt.figure(1)
ax = fig.gca()
ax.plot(x, A)
ax.set_title('A kernel: ' + str(ker))
plt.show()
