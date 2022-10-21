# VKOGA
Python implementation of the Vectorial Kernel Orthogonal Greedy Algorithm.


## Usage:
The algorithm is implemented as a [scikit-learn](https://scikit-learn.org/stable/) `Estimator`, and it can be used via the `fit` and `predict` methods.

The best way to start using the algorithm is having a look at the [demo notebook](demo.ipynb), which can also be executed online on Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GabrieleSantin/VKOGA/master?filepath=demo.ipynb)




## How to cite:
If you use this code in your work, please cite the paper

> G. Santin, B. Haasdonk, Kernel methods for surrogate modeling, In: P. Benner, S. Grivet-
Talocia, A. Quarteroni, G. Rozza, W. Schilders, and L. M. Silveira, editors, Model Order Reduc-
tion, volume 2. De Gruyter, 2021.


```bibtex:
@InCollection{Santin2021,
  author       = {Santin, Gabriele and Haasdonk, Bernard},
  title        = {Kernel Methods for Surrogate Modeling},
  booktitle    = {Model Order Reduction},
  year         = {2021},
  editor       = {Benner, Peter and Grivet-Talocia, Stefano and Quarteroni, Alfio and Rozza, Gianluigi and Schilders, Wil and Silveira, Luís Miguel},
  booksubtitle = {System- and Data-Driven Methods and Algorithms},
  volume       = {2},
  publisher    = {De Gruyter},
}
```

For further details on the algorithm and its implementation, please refer to the following papers:

> M. Pazouki and R. Schaback, [_Bases for kernel-based spaces_](https://www.sciencedirect.com/science/article/pii/S0377042711002688), J. Comput. Appl. Math., 236, 575-588 (2011).

> D. Wirtz and B. Haasdonk, [_A Vectorial Kernel Orthogonal Greedy Algorithm_](https://drna.padovauniversitypress.it/2013/specialissue/10), Dolomites Res. Notes Approx., 6, 83-100 (2013). 

> G. Santin, D. Wittwar, B. Haasdonk, [_Greedy regularized kernel interpolation_](https://arxiv.org/abs/1807.09575), ArXiv preprint 1807.09575 (2018).

> T. Wenzel, G. Santin, B. Haasdonk, [_A novel class of stabilized greedy kernel approximation algorithms: Convergence, stability & uniform point distribution_](https://arxiv.org/abs/1911.04352), ArXiv preprint 1911.04352 (2019).

## Other implementations:
The original Matlab version of this software is maintained [here](https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/vkoga).
