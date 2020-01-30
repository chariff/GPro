import numpy as np


def assert_array(x):
    """Throw a TypeError if X is not array-like."""
    if not (isinstance(x, np.ndarray) or isinstance(x, list)):
        raise TypeError('Only lists and numpy arrays are supported.')


def convert_array(x):
    """Convert a list to a numpy array."""
    if isinstance(x, list):
        # data-type is inferred from the input data.
        x = np.asarray(x)
    return x


def set_d_type(x, d_type):
    """Sets the d_type of a numpy array."""
    if not isinstance(x[0, 0], d_type):
        x = x.astype(d_type)
    return x


def assert_finite(x):
    """Throw a ValueError if x contains NaNs or infinity."""
    if not np.isfinite(x.sum()):
        raise ValueError('Only finite numbers are supported.')


def assert_dim(x):
    """Throw an Assertion error if x is a 1d array."""
    assert len(x.shape) > 1, \
        "Array %r is of inconsistent dimensions." % x


def assert_object(x):
    """Throw a Type error if X is an object."""
    if x.dtype.kind == "O":
        raise TypeError('Object type is not supported for %r.' % x)


def check_x_m(x, m):
    """Input validation for standard estimators.

    Checks x and m for consistent shapes. By default, x and m are
    checked to be non-empty and containing only finite values. m
    is also checked to be containing only positives indexes of x.

    Parameters
    ----------
    x : array-like.
    Input data.

    m : array-like.
    Preferences.

    Returns
    -------
    x : the validated x.

    m : the validated m.
    """

    # Pandas data frame not supported.
    assert_array(x), assert_array(m)
    # If list converts to numpy array.
    x = convert_array(x)
    m = convert_array(m)
    # Check valid dimensions
    assert_dim(x), assert_dim(m)
    # Only real values are supported.
    assert_object(x), assert_object(m)

    if x.dtype.kind not in ('f', 'i', 'u'):
        raise TypeError('Only floating-point, signed or unsigned integer,\
        training data supported.')
    if m.dtype.kind not in ('i', 'u'):
        raise TypeError('Only integer preference data supported.')
    # float64 for x and int8 for m.
    x = set_d_type(x, d_type=np.float64)
    m = set_d_type(m, d_type=np.int8)
    # Only finite numbers are supported.
    assert_finite(x), assert_finite(m)
    # Only positive numbers are supported for preferences.
    if any(m.ravel() < 0):
        raise ValueError('Only positive integers are supported for m.')
    # A preference set should contain two values.
    assert m.shape[1] == 2, \
        "Array %r is of inconsistent dimensions." % m
    assert x.shape[0] > 1, \
        "Array %r is of inconsistent size." % x
    # Check if indexes of m are consistent with size of x.
    if m.max() > x.shape[0]:
        raise ValueError('Preferences should be indexes of X.')
    if any(np.subtract(m[:, 0], m[:, 1]) == 0):
        raise ValueError('m contains at least one set of preferences'
                         ' with the same values.')

    return x, m


def check_post_approx(**params):
    """Input validation for the Laplace approximation.
    Checks s_eval, max_iter, eta, tol for consistent values and shapes.
    """
    s_eval = params['s_eval']
    max_iter = params['max_iter']
    eta = params['eta']
    tol = params['tol']

    if np.isscalar(s_eval) and not isinstance(s_eval, str):
        if max_iter <= 0:
            raise ValueError("s_eval must be a positive scalar.")
    else:
        raise ValueError("s_eval must be a positive scalar.")

    if np.isscalar(max_iter) and not isinstance(max_iter, str):
        if not (isinstance(max_iter, int) and max_iter > 0):
            raise ValueError("max_iter must be a positive integer scalar.")
    else:
        raise ValueError("max_iter must be a positive integer scalar.")

    if np.isscalar(eta) and not isinstance(eta, str):
        if eta < 0:
            raise ValueError("eta must be a positive scalar.")
    else:
        raise ValueError("eta must be a positive scalar.")

    if np.isscalar(tol) and not isinstance(tol, str):
        if tol < 0:
            raise ValueError("tol must be a positive scalar.")
    else:
        raise ValueError("tol must be a positive scalar.")

    return


def check_kernel(x, **params):
    """Input validation for the RBF and Matern kernel.

    Checks length_scale and nu for consistent shape and value.

    Parameters
    ----------
    x : array-like.
    Input data.

    Returns
    -------
    None
    """

    length_scale = params['length_scale']
    if np.iterable(length_scale):
        if np.asarray(length_scale).dtype.kind not in ('f', 'i', 'u'):
            raise TypeError('Only floating-point, signed or unsigned integer,\
            length_scale supported.')
        elif any(length_scale) <= 0:
            raise ValueError("length_scale values must be positive.")
        assert x.shape[0] == len(length_scale), \
            "Array length_scale is of inconsistent dimension."
    elif isinstance(length_scale, str):
        raise ValueError("length_scale must be a positive scalar.")

    if len(params) > 1:
        nu = params['nu']
        if np.isscalar(nu) and not isinstance(nu, str):
            if nu <= 0:
                raise ValueError("nu must be a positive scalar.")
        else:
            raise ValueError("nu must be a positive scalar.")

    return


def check_acquisition(**params):
    """Input validation for acquisition functions.
    Checks kappa and nu for consistent values and shapes.
    """
    key = list(params)[0]
    value = params[key]

    if np.isscalar(value) and not isinstance(value, str):
        if value < 0:
            raise ValueError("%s must be a positive scalar." % key)
    else:
        raise ValueError("%s must be a positive scalar." % key)


def check_bounds(x, bounds):
    """Input validation for .
    Checks kappa and nu for consistent values and shapes.
    """
    if not isinstance(bounds, dict):
        raise TypeError('bounds should be a dictionary')
    assert x.shape[1] == len(bounds), \
        "bounds is of inconsistent size."
    for key_value in bounds.items():
        values = key_value[1]
        if not (isinstance(values, tuple) or isinstance(values, list)):
            raise TypeError('bounds values should be stored in list or tuple')
        assert len(values) == 2, "bounds is of inconsistent size."
        inf, sup = values
        if isinstance(inf, str) or isinstance(sup, str):
            raise ValueError('bounds values should be numeric.')
        assert inf < sup, "inf bound cannot be superior to sup bound."
