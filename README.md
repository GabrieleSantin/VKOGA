# VKOGA
Python implementation of the Vectorial Kernel Orthogonal Greedy Algorithm.


## Usage:
The algorithm is implemented as a [scikit-learn](https://scikit-learn.org/stable/) `Estimator`, and it can be used via the `fit` and `predict` methods.

The best way to start using the algorithm is having a look at the [demo notebook](demo.ipynb). 

The demo can also be executed online on Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/gabriele.santin%2Fvkoga/master?filepath=demo.ipynb)


## How to cite:
If you use this code in your work, please cite the paper

> G. Santin and B. Haasdonk, [_Kernel Methods for Surrogate Modeling_](https://arxiv.org/abs/1907.10556), ArXiv preprint 1907.10556 (2019)


```bibtex:
@TechReport{SaHa2019,
  Author                   = {Santin, Gabriele and Haasdonk, Bernard},
  Title                    = {Kernel Methods for Surrogate Modeling},
  Year                     = {2019},
  Number                   = {1907.10556},
  Type                     = {ArXiv},
  Url                      = {https://arxiv.org/abs/1907.10556}
}
```

For further details on the algorithm and its implementation, please refer to these papers:

> M. Pazouki and R. Schaback, [_Bases for kernel-based spaces_](https://www.sciencedirect.com/science/article/pii/S0377042711002688), J. Comput. Appl. Math., 236, 575-588 (2011).

> D. Wirtz and B. Haasdonk, [_A Vectorial Kernel Orthogonal Greedy Algorithm_](https://drna.padovauniversitypress.it/2013/specialissue/10), Dolomites Res. Notes Approx., 6, 83-100 (2013). 

> G. Santin, D. Wittwar, B. Haasdonk, [_Greedy regularized kernel interpolation_](https://arxiv.org/abs/1807.09575), ArXiv preprint 1807.09575 (2018).

> T. Wenzel, G. Santin, B. Haasdonk, [_A novel class of stabilized greedy kernel approximation algorithms: Convergence, stability & uniform point distribution_](https://arxiv.org/abs/1911.04352), ArXiv preprint 1911.04352 (2019).

## Other implementations:
The original Matlab version of this software is maintained here:
[VKOGA](https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/vkoga).
