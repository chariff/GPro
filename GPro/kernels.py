from inspect import signature
import math
import numpy as np
from scipy import special
from scipy.spatial.distance import cdist, pdist, squareform

# Note: this module is strongly inspired by the gaussian process module
# of the sklearn-package courtesy Jan Hendrik Metzen.


class Kernel:
    """Base class for all kernels.

    .. versionadded:: 0.1
    """
    def get_params(self):
        """Get the parameters of a kernel.

         Returns
        -------
        params : mapping of string to any parameter
        names mapped to their values.
        """
        params = dict()
        cls = self.__class__
        init_sign = signature(cls)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if (parameter.kind != parameter.VAR_KEYWORD and
                    parameter.name != 'self'):
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)
        if len(varargs) != 0:
            raise RuntimeError("GPro kernels should always "
                               " specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls,))
        for arg in args:
            params[arg] = getattr(self, arg, None)
        return params

    def set_params(self, **params):
        """Set the parameters of a kernel.

        Returns
        -------
        self
        """
        if not params:
            return self
        valid_params = self.get_params()
        for key, value in params.items():
            # simple objects case
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for kernel %s. '
                                 'Check the list of available parameters '
                                 'with `Kernel.get_params().keys()`.' %
                                 (key, self.__class__.__name__))
            setattr(self, key, value)
        return self


class RBF(Kernel):
    """Radial-basis function kernel.

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length-scale
    parameter length_scale>0, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel).

    Parameters
    ----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    """

    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    def __call__(self, X, Y=None):
        """Return the kernel k(X, Y).

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        """

        if Y is None:
            dists = pdist(X / self.length_scale, metric='sqeuclidean')
        else:
            dists = cdist(X / self.length_scale, Y / self.length_scale,
                          metric='sqeuclidean')

        K = np.exp(-.5 * dists)
        # convert from upper-triangular matrix to square matrix
        if Y is None:
            K = squareform(K)
            np.fill_diagonal(K, 1)
        return K


class Matern(RBF):
    """ Matern kernel.

    The class of Matern kernels is a generalization of the RBF and the
    absolute exponential kernel parameterized by an additional parameter
    nu. The smaller nu, the less smooth the approximated function is.
    For nu=inf, the kernel becomes equivalent to the RBF kernel and for nu=0.5
    to the absolute exponential kernel. Important intermediate values are
    nu=1.5 (once differentiable functions) and nu=2.5 (twice differentiable
    functions).

    See Rasmussen and Williams 2006, pp84 for details regarding the
    different variants of the Matern kernel.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    nu : float, default: 1.5
        The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.

    """

    def __init__(self, length_scale=1.0, nu=1.5):
        super().__init__(length_scale)
        self.nu = nu

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    def __call__(self, X, Y=None):
        """Return the kernel k(X, Y).

        Parameters
        ----------

        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.


        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        """
        if Y is None:
            dists = pdist(X / self.length_scale, metric='euclidean')
        else:
            dists = cdist(X / self.length_scale, Y / self.length_scale,
                          metric='euclidean')
        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1. + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = (math.sqrt(2 * self.nu) * K)
            K.fill((2 ** (1. - self.nu)) / special.gamma(self.nu))
            K *= tmp ** self.nu
            K *= special.kv(self.nu, tmp)
        if Y is None:
            K = squareform(K)
            np.fill_diagonal(K, 1)
        return K
