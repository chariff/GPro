[![Build Status](https://travis-ci.org/chariff/GPro.svg?branch=master)](https://travis-ci.org/chariff/GPro)
[![Codecov](https://codecov.io/github/chariff/GPro/badge.svg?branch=master&service=github)](https://codecov.io/github/chariff/GPro?branch=master)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

Python implementation of a probabilistic kernel approach to preference 
learning based on Gaussian processes. Preference relations are captured 
in a Bayesian framework which allows in turn for global optimization of 
the inferred functions (Gaussian processes) in as few iterations as possible. 

Brief guide to using GPro
=========================

## 1. Fitting and making Predictions.

```python
from GPro.preference import ProbitPreferenceGP
import numpy as np

# Training data consisting of numeric real positive values.
# A minimum of two values is required.
X = np.array([[2], [1]]).reshape(-1, 1)
# Target choices. A preference is an array of positive
# integers of shape = (2,). preference[0], is an index
# of X preferred over preference[1], which is an index of X.
M = np.array([0, 1]).reshape(-1, 2)
# Instantiate a ProbitPreferenceGP object with default parameters.
gpr = ProbitPreferenceGP()
# Fit a Gaussian process. A flat prior with mean zero is applied by default.
gpr.fit(X, M, f_prior=None)
# Predict values.
gpr.predict(X, return_y_var=True)
>>> (array([[ 0.22129742],
>>>        [-0.22129742]]), array([[0.00316225],
>>>        [0.00316225]]))
```


## 2. Probit Bayesian optimization via preferences inputs.

```python
from GPro.kernels import Matern
from GPro.posterior import Laplace
from GPro.acquisitions import UCB
from GPro.optimization import ProbitBayesianOptimization
import numpy as np


# Training data consisting of numeric real positive values.
# A minimum of two values is required.
X = np.random.sample(size=(2, 3)) * 10
# Target choices. A preference is an array of positive
# integers of shape = (2,). preference[0], is an index
# of X preferred over preference[1], which is an index of X.
M = np.array([0, 1]).reshape(-1, 2)
# Parameters for the ProbitBayesianOptimization object.
GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
             'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
                                    eta=0.01, tol=1e-3),
             'acquisition': UCB(kappa=2.576),
             'alpha': 1e-5,
             'random_state': 2020}
# instantiate a ProbitBayesianOptimization object with custom parameters.
gpr_opt = ProbitBayesianOptimization(X, M, GP_params)
# Bounded region of optimization space.
bounds = {'x0': (0, 10), 'x1': (0, 10), 'x2': (0, 10)}
# Console optimization method.
console_opt = gpr_opt.console_optimization(bounds=bounds, n_init=100, n_solve=10)
optimal_values, X_post, M_post, f_post = console_opt
print('optimal values: ', optimal_values)

# Use posterior as prior
# PreferenceBayesianOptimization
gpr_opt = ProbitBayesianOptimization(X_post, M_post, GP_params)
console_opt = gpr_opt.console_optimization(bounds=bounds, n_init=100, n_solve=10,
                                           f_prior=f_post)

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
                  x0        x1        x2
>>> preference  1.083927  4.273101  9.905242
>>> suggestion  1.570381  5.079068  8.668470
>>> Iteration 2, preference (p) or suggestion (s)? (Q to quit): Q
```

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


# function optimization example.
def random_sample(n, d, bounds, random_state=None):
    # Uniform sampling given bounds.
    if random_state is None:
        random_state = np.random.randint(1e6)
    random_state = np.random.RandomState(random_state)
    sample = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                  size=(n, d))
    return sample


def sample_normal_params(n, d, bounds, scale_sigma=1, random_state=None):
    # Sample parameters of a multivariate normal distribution
    # sample centroids.
    mu = random_sample(n=n, d=d, bounds=np.array(list(bounds.values())),
                       random_state=random_state)
    # sample covariance matrices.
    sigma = datasets.make_spd_matrix(d, random_state) * scale_sigma
    theta = {'mu': mu, 'sigma': sigma}
    return theta


d = 2
bounds = {'x' + str(i): (0, 10) for i in range(0, d)}
theta = sample_normal_params(n=1, d=d, bounds=bounds, scale_sigma=10, random_state=12)
f = lambda x: multivariate_normal.pdf(x, mean=theta['mu'][0], cov=theta['sigma'])
# X, M, init
X = random_sample(n=2, d=d, bounds=np.array(list(bounds.values())))
X = np.asarray(X, dtype='float64')
M = sorted(range(len(f(X))), key=lambda k: f(X)[k], reverse=True)
M = np.asarray([M], dtype='int8')
GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
             'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
                                    eta=0.01, tol=1e-3),
             'acquisition': UCB(kappa=2.576),
             'alpha': 1e-5,
             'random_state': 2020}
gpr_opt = ProbitBayesianOptimization(X, M, GP_params)
function_opt = gpr_opt.function_optimization(f=f, bounds=bounds, max_iter=d*10,
                                             n_init=1000, n_solve=1)

optimal_values, X_post, M_post, f_post = function_opt
print('optimal values: ', optimal_values)

# rmse
print('rmse: ', .5 * sum(np.sqrt((optimal_values - theta['mu'][0]) ** 2)))
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

![](https://github.com/chariff/GPro/examples/mvn_example.png)