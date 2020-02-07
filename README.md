

# Preference learning with gaussian processes.

[![Build Status](https://travis-ci.org/chariff/GPro.svg?branch=master)](https://travis-ci.org/chariff/GPro)
[![Codecov](https://codecov.io/github/chariff/GPro/badge.svg?branch=master&service=github)](https://codecov.io/github/chariff/GPro?branch=master)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)


Python implementation of a probabilistic kernel approach to preference 
learning based on Gaussian processes. Preference relations are captured 
in a Bayesian framework which allows in turn for global optimization of 
the inferred functions (Gaussian processes) in as few iterations as possible.

Installation
============

### Installation
* From PyPI:

      pip install GPro

* From GitHub:

      pip install git+https://github.com/chariff/GPro.git

### Dependencies
GPro requires:
* Python (>= 3.5)
* NumPy (>= 1.9.0)
* SciPy (>= 0.14.0)
* Pandas (>= 0.24.0) 

Brief guide to using GPro.
=========================

Checkout the package docstrings for more informations.

## 1. Fitting and making Predictions.

```python
from GPro.preference import ProbitPreferenceGP
import numpy as np
```
Training data consisting of numeric real positive values.
A minimum of two values is required.
```python
X = np.array([[2], [1]]).reshape(-1, 1)
```
M is an array containing preferences. A preference is an 
array of positive integers of shape = (2,). The left integer of a 
given preference is the index of a value in X which is preferred 
over another value of X indexed by the right integer of the same
preference array.
In the following example, 2 is preferred over 1.
```python
M = np.array([0, 1]).reshape(-1, 2)
```
Instantiate a ProbitPreferenceGP object with default parameters.
```python
gpr = ProbitPreferenceGP()
```
Fit a Gaussian process. A flat prior with mean zero is applied by default.
```python
gpr.fit(X, M, f_prior=None)
```
Predict new values.
```python
X_new = np.linspace(-6, 9, 100).reshape(-1, 1)
predicted_values, predicted_vars = gpr.predict(X_new, return_y_var=True)
```
Plot.
```python
plt.plot(X_new, np.zeros(100), 'k--', label='GP prior')
plt.plot(X_new, predicted_values, 'r-', label='GP posterior')
plt.plot(X.flat, gpr.predict(X).flat, 'bx', label='Preference')
plt.ylabel('f(X)')
plt.xlabel('X')
plt.gca().fill_between(X_new.flatten(),
                       (predicted_values - predicted_vars).flatten(),
                       (predicted_values + predicted_vars).flatten(),
                       color="#b0e0e6", label='GP posterior s.d.')
plt.legend()
plt.ylim([-2, 2])
plt.show()
```
The following plot shows how the posterior gaussian process is adjusted to 
the data i.e. 2 is preferred to 1. One can also notice how the standard 
deviation is small where there is data.  

![Gaussian process posterior](https://github.com/chariff/GPro/blob/master/examples/posterior_example.png)

## 2. Interactive bayesian optimization.

Preference relations are captured in a Bayesian framework 
which allows for global optimization of the latent function 
(modelized by gaussian processes) describing the preference relations.
Interactive bayesian optimization with probit responses works by querying
the user with a paired comparison and by subsequently updating the 
Gaussian process model. The iterative procedure optimizes a utility function,
seeking a balance between exploration and exploitation of the latent function, 
to present the user with a new set of instances.
```python
from GPro.kernels import Matern
from GPro.posterior import Laplace
from GPro.acquisitions import UCB
from GPro.optimization import ProbitBayesianOptimization
import numpy as np

# 3D example. Initialization.
X = np.random.sample(size=(2, 3)) * 10
M = np.array([0, 1]).reshape(-1, 2)
```
Custom parameters for the ProbitBayesianOptimization object. 
Checkout the package docstrings for more informations on the parameters.
```python
GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
             'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
                                    eta=0.01, tol=1e-3),
             'acquisition': UCB(kappa=2.576),
             'alpha': 1e-5,
             'random_state': 2020}
```
Instantiate a ProbitBayesianOptimization object with custom parameters.
```python
gpr_opt = ProbitBayesianOptimization(X, M, GP_params)
```
Bounded region of optimization space.
```python
bounds = {'x0': (0, 10), 'x1': (0, 10), 'x2': (0, 10)}
```
Interactive optimization method.
Checkout the package docstrings for more informations on the parameters.
```python
console_opt = gpr_opt.interactive_optimization(bounds=bounds, n_init=100, n_solve=10)
optimal_values, suggestion, X_post, M_post, f_post = console_opt
print('optimal values: ', optimal_values)
```
    >>>                   x0        x1        x2
    >>> preference  0.806058  5.567739  9.924089
    >>> suggestion  0.420045  7.317108  9.950919
    >>> Iteration 0, preference (p) or suggestion (s)? (Q to quit): p
    >>>                   x0        x1        x2
    >>> preference  0.806058  5.567739  9.924089
    >>> suggestion  1.083927  4.273101  9.905242
    >>> Iteration 1, preference (p) or suggestion (s)? (Q to quit): s
    >>>                   x0        x1        x2
    >>> preference  1.083927  4.273101  9.905242
    >>> suggestion  1.570381  5.079068  8.668470
    >>> Iteration 2, preference (p) or suggestion (s)? (Q to quit): Q
    >>> optimal values:  [1.08392668 4.27310139 9.90524192]
One can use informative prior. Let's use posterior as prior for the sake of
example.
```python
gpr_opt = ProbitBayesianOptimization(X_post, M_post, GP_params)
console_opt = gpr_opt.interactive_optimization(bounds=bounds, n_init=100, 
                                               n_solve=10, f_prior=f_post,
                                               max_iter=1, print_suggestion=False)
optimal_values, suggestion, X_post, M_post, f_post = console_opt
```

## 2. Bayesian optimization of a black-box function.

**Disclaimer:** For testing purposes, we maximize a multivariate normal pdf.
```python
from GPro.kernels import Matern
from GPro.posterior import Laplace
from GPro.acquisitions import UCB
from GPro.optimization import ProbitBayesianOptimization
from scipy.stats import multivariate_normal
import numpy as np
from sklearn import datasets
import matplotlib.cm as cm
import matplotlib.pyplot as plt
```
Uniform sampling given bounds.
```python
def random_sample(n, d, bounds, random_state=None):
    if random_state is None:
        random_state = np.random.randint(1e6)
    random_state = np.random.RandomState(random_state)
    sample = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                  size=(n, d))
    return sample
```
Sample parameters of a multivariate normal distribution
```python
def sample_normal_params(n, d, bounds, scale_sigma=1, random_state=None):
    # sample centroids.
    mu = random_sample(n=n, d=d, bounds=np.array(list(bounds.values())),
                       random_state=random_state)
    # sample covariance matrices.
    sigma = datasets.make_spd_matrix(d, random_state) * scale_sigma
    theta = {'mu': mu, 'sigma': sigma}
    return theta
```
Example is in 2 dimensions.
```python
d = 2
# Bounded region of optimization space.
bounds = {'x' + str(i): (0, 10) for i in range(0, d)}
# Sample parameters of a d-multivariate normal distribution
theta = sample_normal_params(n=1, d=d, bounds=bounds, scale_sigma=10, random_state=12)
# function to be optimized.
f = lambda x: multivariate_normal.pdf(x, mean=theta['mu'][0], cov=theta['sigma'])
# X, M, init
X = random_sample(n=2, d=d, bounds=np.array(list(bounds.values())))
X = np.asarray(X, dtype='float64')
# Target choices. A preference is an array of positive
# integers of shape = (2,). preference[0], is an index
# of X preferred over preference[1], which is an index of X.
M = sorted(range(len(f(X))), key=lambda k: f(X)[k], reverse=True)
M = np.asarray([M], dtype='int8')
# Parameters for the ProbitBayesianOptimization object.
GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
             'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
                                    eta=0.01, tol=1e-3),
             'acquisition': UCB(kappa=2.576),
             'alpha': 1e-5,
             'random_state': 2020}
# instantiate a ProbitBayesianOptimization object with custom parameters
gpr_opt = ProbitBayesianOptimization(X, M, GP_params)
```
Function optimization method.
```python
function_opt = gpr_opt.function_optimization(f=f, bounds=bounds, max_iter=d*10,
                                             n_init=1000, n_solve=1)

optimal_values, X_post, M_post, f_post = function_opt
print('optimal values: ', optimal_values)
```
    >>> optimal values:  [1.45340052 7.22687626]
```python
# rmse
print('rmse: ', .5 * sum(np.sqrt((optimal_values - theta['mu'][0]) ** 2)))
```
    >>> rmse:  0.13092430596422377
```python
# 2d plot
if d == 2:
    resolution = 10
    x_min, x_max = bounds['x0'][0], bounds['x0'][1]
    y_min, y_max = bounds['x1'][0], bounds['x1'][1]
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    grid = np.empty((resolution ** 2, 2))
    grid[:, 0] = X.flat
    grid[:, 1] = Y.flat
    Z = f(grid)
    plt.imshow(Z.reshape(-1, resolution), interpolation="bicubic",
               origin="lower", cmap=cm.rainbow, extent=[x_min, x_max, y_min, y_max])
    plt.scatter(optimal_values[0], optimal_values[1], color='black', s=10)
    plt.title('Target function')
    plt.colorbar()
    plt.show()
```

![](https://github.com/chariff/GPro/blob/master/examples/mvn_example.png)

### References:
* http://mlg.eng.cam.ac.uk/zoubin/papers/icml05chuwei-pl.pdf
* https://arxiv.org/pdf/1012.2599.pdf
* https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf
* http://www.gaussianprocess.org/gpml/


    -- Chariff Alkhassim