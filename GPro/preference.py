# Author: Chariff Alkhassim <chariff.alkhassim@gmail.com>
# License: MIT

# Note: this module is inspired by the gaussian process module
# of the sklearn-package and by the util module of
# the BayesianOptimization package. Courtesy of Jan Hendrik Metzen
# and Fernando M. F. Nogueira.


from .kernels import Kernel, RBF
from .posterior import Laplace
from .acquisitions import Acquisition, ExpectedImprovement
from scipy.optimize import minimize
import numpy as np
import warnings
from .validations import assert_array,  assert_finite, check_x_m, \
    check_kernel, check_post_approx, check_acquisition,\
    check_bounds, convert_array


class ProbitPreferenceGP(Kernel, Acquisition):
    """ Probabilistic kernel approach to preference learning.

    The implementation is based on Preference Learning with Gaussian Processes
    by Chu and Ghahramani 2006.

    .. versionadded:: 0.1

    Parameters
    ----------
    kernel : object
        The kernel specifying the covariance function of the GP. If None is
        passed, the rbf kernel is used as default with gamma=1.
        Note that the kernel's hyperparameters are optimized during fitting.

    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive-definite matrix.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=alpha.
        Allowing to specify the noise level directly as a parameter is mainly
        for convenience and for consistency with Ridge.

    post_approx : object
        The posterior approximation method. If None is passed, the Laplace
        approximation is used as default.

    acquisition : object
        The acquisition function enabling an optimization procedure
        to sample an optimal point based on attributes of the
        posterior distribution. If None is passed, the expected improvement
        is used as default.

    copy_data : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.


    Attributes
    ----------

    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (required for prediction)

    M_train_ : array-like, shape = (n_samples - 1, 2)
            Target choices. A preference is an array of positive
            integers of shape = (2,). preference[0], r, is an index
            of X preferred over preference[1], c, which is an
            index of X. (required for prediction)

    kernel_ : kernel object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters

    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``

    f_posterior_  : array-like, shape = (n_samples, 1)
        Posterior distribution of the  Gaussian Process.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from GPro.preference import ProbitPreferenceGP
    >>> import numpy as np

    >>> X = np.array([[2], [1]])
    >>> M = np.array([[0, 1]])
    >>> gpr = ProbitPreferenceGP()
    >>> gpr.fit(X, M)
    >>> X_new = np.linspace(-6, 9, 100).reshape(-1, 1)
    >>> predicted_values, predicted_vars = gpr.predict(X_new, return_y_std=True)
    >>> plt.plot(X_new, np.zeros(100), 'k--', label='GP prior')
    >>> plt.plot(X_new, predicted_values, 'r-', label='GP posterior')
    >>> plt.plot(X.flat, gpr.predict(X).flat, 'bx', label='Preference')
    >>> plt.ylabel('f(X)')
    >>> plt.xlabel('X')
    >>> plt.gca().fill_between(X_new.flatten(),
                           (predicted_values - predicted_vars).flatten(),
                           (predicted_values + predicted_vars).flatten(),
                           color="#b0e0e6", label='GP posterior s.d.')

    >>> plt.legend()
    >>> plt.ylim([-2, 2.5])
    >>> plt.show()
    """

    def __init__(self, kernel=None, alpha=1e-5,
                 post_approx=None, acquisition=None,
                 copy_data=True, random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.post_approx = post_approx
        self.copy_data = copy_data
        self.acquisition = acquisition
        self.random_state = random_state

    def fit(self, X, M, f_prior=None):
        """Fit a Gaussian process probit regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data consisting of numeric real positive values.

        M : array-like, shape = (n_samples - 1, 2)
            Target choices. A preference is an array of positive
            integers of shape = (2,). preference[0], r, is an index
            of X preferred over preference[1], c, which is an
            index of X.

        f_prior : array-like, shape = (n_samples, 1), optional (default: None)
            Flat prior with mean zero is applied by default.


        Returns
        -------
        self : returns an instance of self.
        """

        X, M = check_x_m(X, M)
        self.X_train_ = np.copy(X) if self.copy_data else X
        self.M_train_ = np.copy(M) if self.copy_data else M

        if self.kernel is None:  # Use a RBF kernel as default
            self.kernel = RBF(length_scale=1.0)
            self.kernel_ = self.kernel
        else:
            if not hasattr(self.kernel, "get_params"):
                raise AttributeError("Invalid kernel.")
            self.kernel_ = self.kernel
            check_kernel(X, **self.kernel.get_params())

        if self.post_approx is None:  # Use a Laplace approximation
            self.post_approx = Laplace(s_eval=self.alpha, max_iter=1000,
                                       eta=0.01, tol=1e-5)
        else:
            if not hasattr(self.post_approx, "get_params"):
                raise AttributeError("Invalid post_approx.")
            check_post_approx(**self.post_approx.get_params())

        if np.iterable(self.alpha):
            self.alpha = convert_array(self.alpha)
            if self.alpha.shape[0] != self.X_train_.shape[0]:
                if self.alpha.shape[0] == 1:
                    self.alpha = self.alpha[0]
                else:
                    raise ValueError("alpha must be a scalar or an array"
                                     " with same number of entries as X.(%d != %d)"
                                     % (self.alpha.shape[0], X.shape[0]))
        elif not np.isscalar(self.alpha):
            raise ValueError("alpha must be a scalar or an array"
                             " with same number of entries as X.(%d != %d)"
                             % (self.alpha.shape[0], X.shape[0]))

        # compute quantities required for prediction
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive-definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "ProbitPreferenceGP estimator."
                        % self.kernel_,) + exc.args
            raise
        if f_prior is None:
            # flat f_prior with mean zero
            f_prior = np.zeros((self.L_.shape[0], 1))
        else:
            assert_array(f_prior)
            assert_finite(f_prior)
            if f_prior.dtype.kind not in ('f', 'i', 'u'):
                raise TypeError('Only floating-point, signed or unsigned integer,\
                prior data supported.')

        # compute the posterior distribution of f
        self.f_posterior_, c = self.post_approx(f=f_prior, M=self.M_train_, K=K)

        # precompute part of the posterior covariance matrix
        # (c K + I)^-1 c, where c is the nll Hessian
        self.K_cov_inv = np.linalg.solve(c @ K + np.identity(K.shape[0]), c)

        return self

    @property
    def posterior(self):
        """Returning the posterior distribution of f."""
        if not hasattr(self, "f_posterior_"):
            raise AttributeError("Unfitted gaussian probit regression model.")
        return self.f_posterior_

    def predict(self, X, return_y_std=False, return_y_cov=False):
        """Predict using the Gaussian process regression model

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        return_y_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_y_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_y_std` is True.

        y_cov : ndarray of shape (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_y_cov` is True.
        """
        if not hasattr(self, "X_train_"):
            raise AttributeError("Unfitted gaussian probit regression model.")

        if return_y_std and return_y_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        K_trans = self.kernel_(self.X_train_, X)
        Lk = np.linalg.solve(self.L_, K_trans)
        Lf = np.linalg.solve(self.L_, self.f_posterior_)
        y_mean = np.dot(Lk.T, Lf)

        if return_y_cov:
            y_cov = self.kernel_(X) - K_trans.T @ self.K_cov_inv @ K_trans

            return y_mean, y_cov
        elif return_y_std:
            # faster computation of the diagonal of `y_cov`
            # mimics https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/gaussian_process/_gpr.py#L368
            y_var = np.diag(self.kernel_(X)) - \
                np.einsum('ij,jk,ki->i', K_trans.T, self.K_cov_inv, K_trans)
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0

            return y_mean, np.sqrt(y_var).reshape(-1, 1)
        else:
            return y_mean

    def bayesopt(self, bounds, method="L-BFGS-B", n_init=1, n_solve=1):
        """Bayesian optimization based on the optimization of a
        utility function of the attributes of the posterior distribution.

        Parameters
        ----------
        bounds: dictionary
        Bounds of the search space for the acquisition function.

        method: str or callable, optional
        Type of solver.

        n_init: integer, optional
            Number of initialization points for the solver. Obtained
            by randomly sampling the acquisition function.

        n_solve: integer, optional
            The solver will be run n_solve times.
            Cannot be superior to n_init.

        Returns
        -------
        sample_opt : array, shape = (1, [n_output_dims])
            Sample point.

        """

        if not hasattr(self, "f_posterior_"):
            raise AttributeError("Unfitted gaussian probit regression model.")
        if n_init < n_solve:
            raise ValueError("n_init must be inferior of equal to n_solve.")
        check_bounds(self.X_train_, bounds)
        # convert bounds values to ndarray
        bounds = np.array(list(bounds.values()))
        if self.acquisition is None:  # Use an ei acquisition as default
            acquisition = ExpectedImprovement(xi=0)
            self.acquisition = acquisition
        else:
            if not hasattr(self.acquisition, "get_params"):
                raise AttributeError("Invalid acquisition.")
            acquisition = self.acquisition
            check_acquisition(**acquisition.get_params())

        y_max = self.f_posterior_.max()

        # Warm up with random points
        def random_sample(d, bounds, n, random_state):
            if random_state is None:
                random_state = np.random.randint(1e6)
            random_state = np.random.RandomState(random_state)
            samples = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                           size=(n, d))
            return samples

        d = self.X_train_.shape[1]
        x_tries = random_sample(d=d, bounds=bounds, n=n_init,
                                random_state=self.random_state)

        def aqc_optim(x, y_max):
            y_mean, std = self.predict(X=x, return_y_std=True)
            ys = acquisition(y_mean, std, y_max)
            return ys

        ys = aqc_optim(x_tries, y_max)
        x_arg_max = ys.argmax()
        x_max = x_tries[x_arg_max].reshape(1, -1)
        max_acq = ys.max()
        x_seeds = x_tries[np.argsort(ys.flat)][:n_solve]
        for x_try in x_seeds:
            # Find the minimum of -1 * acquisition function
            res = minimize(lambda x: -aqc_optim(x.reshape(1, -1),
                                                y_max=y_max).ravel(),
                           x_try,
                           bounds=bounds,
                           method=method)
            if not res.success:
                continue
            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]
        return np.clip(x_max, bounds[:, 0], bounds[:, 1]).reshape(1, d)
