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
X = np.array([[2], [1]]).reshape(-1, 1)
# Target choices. A preference is an array of positive
# integers of shape = (2,). preference[0], is an index
# of X preferred over preference[1], which is an index of X.
M = np.array([0, 1]).reshape(-1, 2)
# instantiate a ProbitPreferenceGP object.
gpr = ProbitPreferenceGP()
# Fit a Gaussian process. A flat prior with mean zero is applied by default.
gpr.fit(X, M, f_prior=None)
# Predict values.
print(gpr.predict(X, return_y_var=True))
```

## 2. Bayesian optimization via preferences inputs.