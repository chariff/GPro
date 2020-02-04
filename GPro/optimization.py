import pandas as pd
import numpy as np
from .preference import ProbitPreferenceGP
from .validations import check_x_m


class GProOptimization(ProbitPreferenceGP):
    """

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Training data consisting of numeric real positive values.

    M : array-like, shape = (n_samples, n_preferences)
        Target choices. A preference is an array of positive
        integers of shape = (2,). preference[0], r, is an index
        of X preferred over preference[1], c, which is an
        index of X.
    """
    def __init__(self, X, M, GP_params={}):
        super().__init__(**GP_params)
        self.X = X
        self.M = M

    def console_optimization(self, bounds, method="L-BFGS-B",
                             warm_up=1, n_iter=1, f_prior=None, max_iter=1e4):
        """Bayesian optimization via preferences inputs.

        Parameters
        ----------
        bounds: dictionary
            Bounds of the search space for the acquisition function.

        method: str or callable, optional
            Type of solver.

        warm_up: integer, optional
            Number of times to randomly sample the acquisition function

        n_iter: integer, optional
            Maximum number of iterations to be performed by the solver.

        f_prior : array-like, shape = (n_samples, 1), optional (default: None)
            Flat prior with mean zero is applied by default.

        max_iter: integer, optional (default: 1e4)
            Maximum number of iterations to be performed
            for the bayesian optimization.

        Returns
        -------
        optimal_values : array-like, shape = (n_features, )

        X : array-like, shape = (n_samples, n_features)
            Feature values in training data.

        M : array-like, shape = (n_samples - 1, 2)
            Target choices. A preference is an array of positive
            integers of shape = (2,). preference[0], r, is an index
            of X preferred over preference[1], c, which is an
            index of X.

        f_posterior  : array-like, shape = (n_samples, 1)
            Posterior distribution of the  Gaussian Process.

        Examples
        --------
        >>> from GPro.kernels import Matern
        >>> from GPro.posterior import Laplace
        >>> from GPro.acquisitions import UCB
        >>> from GPro.optimization import GProOptimization
        >>> import numpy as np

        >>> GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
        ...      'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
        ...                             eta=0.01, tol=1e-3),
        ...      'acquisition': UCB(kappa=2.576),
        ...      'random_state': None}
        >>> X = np.array([[2], [1]]).reshape(-1, 1)
        >>> M = np.array([0, 1]).reshape(-1, 2)
        >>> gpr_opt = GProOptimization(X, M, GP_params)
        >>> bounds = {'x0': (0, 10)}
        >>> console_opt = gpr_opt.console_optimization(bounds=bounds, n_iter=1,
        ...                                            warm_up=100)
        >>> optimal_values, X_post, M_post, f_post = console_opt
        >>> print('optimal values: ', optimal_values)

        >>> # Use posterior as prior
        >>> gpr_opt = GProOptimization(X_post, M_post, GP_params)
        >>> console_opt = gpr_opt.console_optimization(bounds=bounds, n_iter=1,
        ...                                            warm_up=100,
        ...                                            f_prior=f_post)
        >>> optimal_values, X_post, M_post, f_post = console_opt
        >>> print('optimal values: ', optimal_values)

        """

        X, M = check_x_m(self.X, self.M)
        features = list(bounds.keys())
        M_ind_cpt = M.shape[0] - 1
        pd.set_option('display.max_rows', None)
        iteration = 0
        while iteration < max_iter:
            self.fit(X, M, f_prior)
            x_optim = self.bayesopt(bounds, method, warm_up, n_iter)
            f_optim = self.predict(x_optim)
            f_prior = np.concatenate((self.posterior, f_optim))
            X = np.concatenate((X, x_optim))
            # current preference index in X.
            M_ind_current = M[M.shape[0] - 1][0]
            # suggestion index in X.
            M_ind_proposal = M_ind_cpt + 2
            # current preference vs suggestion.
            df = pd.DataFrame(data=np.concatenate((X[[M_ind_current]],
                                                   X[[M_ind_proposal]])).T,
                              columns=['preference', 'suggestion'],
                              index=features)
            print(df)
            input_msg = "Iteration %d, preference or suggestion? (Q to quit): " \
                        % M_ind_cpt
            preference_input = input(input_msg)
            if preference_input == 'Q':
                break
            # left index is preferred over right index as a convention.
            elif preference_input == 'p':
                new_pair = np.array([M_ind_current, M_ind_proposal])
            elif preference_input == 's':
                new_pair = np.array([M_ind_proposal, M_ind_current])
            else:
                break
            M = np.vstack((M, new_pair))
            M_ind_cpt += 1
            iteration += 1
        pd.set_option('display.max_rows', 0)
        optimal_values = df['preference'].values
        f_posterior = f_prior
        return optimal_values, X, M, f_posterior

    def function_optimization(self, f, bounds, max_iter=1,
                              method="L-BFGS-B", warm_up=100, n_iter=1,
                              f_prior=None):
        """Bayesian optimization via function evaluation.

        Parameters
        ----------
        f: function object
            A function to be optimized.

        bounds: dictionary
            Bounds of the search space for the acquisition function.

        max_iter: integer, optional
            Maximum number of iterations to be performed
            for the bayesian optimization.

        method: str or callable, optional
            Type of solver.

        warm_up: integer, optional
            Number of times to randomly sample the acquisition function

        n_iter: integer, optional
            Maximum number of iterations to be performed by the solver.

        f_prior : array-like, shape = (n_samples, 1), optional (default: None)
            Flat prior with mean zero is applied by default.

        Returns
        -------
        optimal_values : array-like, shape = (n_features, )

        X : array-like, shape = (n_samples, n_features)
            Feature values in training data.

        M : array-like, shape = (n_samples - 1, 2)
            Target choices. A preference is an array of positive
            integers of shape = (2,). preference[0], r, is an index
            of X preferred over preference[1], c, which is an
            index of X.

        f_posterior  : array-like, shape = (n_samples, 1)
            Posterior distribution of the  Gaussian Process.

        Examples
        --------
        >>> from GPro.kernels import Matern
        >>> from GPro.posterior import Laplace
        >>> from GPro.acquisitions import UCB
        >>> from GPro.optimization import GProOptimization
        >>> from scipy.stats import multivariate_normal
        >>> import numpy as np
        >>> import matplotlib.cm as cm
        >>> import matplotlib.pyplot as plt

        >>> # function optimization example.
        >>> def random_sample(d, bounds, n, random_state=None):
        ...     # Uniform sampling given bounds.
        ...     if random_state is None:
        ...        random_state = np.random.randint(1e6)
        ...     random_state = np.random.RandomState(random_state)
        ...     sample = random_state.uniform(bounds[:, 0], bounds[:, 1],
        ...                                   size=(n, d))
        ...     return sample


        >>> def sample_gm_centers(d, bounds, n_class=2, scale_covar=1,
        ...                       target_p=.9, random_state=None):
        ...     # Sample parameters of a gaussian mixture.
        ...     # sample centroids.
        ...     centroids = random_sample(d=d, bounds=np.array(list(bounds.values())),
        ...                              n=n_class, random_state=random_state)
        ...     # sample covariance matrices.
        ...     # from sklearn import datasets
        ...     # covars = datasets.make_spd_matrix(d, random_state=None)
        ...     covars = np.eye(d) * scale_covar
        ...     # sample mixture distribution conditionally to target_p.
        ...     target_p = np.array([1]) if n_class == 1 else np.array(target_p)
        ...     a, b = 0, (1 - target_p[0])
        ...     left_p = a + (b - a) * np.random.rand(n_class - 1)
        ...     probs = np.concatenate((target_p, (left_p / (left_p.sum()) * b)))
        ...     theta = {'mu': centroids, 'covar': covars, 'probs': probs}
        ...     return theta


        >>> def gm(x, theta):
        ...     # Gaussian mixture pdf.
        ...     centroids, covar = theta['mu'], theta['covar']
        ...     probs = theta['probs']
        ...     n = x.shape[0]
        ...     comp = np.zeros(n)
        ...     for k, p in enumerate(probs, 0):
        ...         comp += p * multivariate_normal.pdf(x, centroids[k], cov=covar)
        ...     return comp


        >>> GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
        ...              'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
        ...                                     eta=0.01, tol=1e-3),
        ...              'acquisition': UCB(kappa=2.576),
        ...              'random_state': None}
        >>> d = 2
        >>> bounds = {'x'+str(i): (0, 10) for i in range(0, d)}
        >>> n_class = 2
        >>> target_p = [.7]
        >>> scale_covar = 2
        >>> theta = sample_gm_centers(d, bounds, n_class, scale_covar,
                                  target_p, random_state=3)
        >>> f = lambda x: gm(x, theta)
        >>> # X, M, init
        >>> X = random_sample(d, np.array(list(bounds.values())), n=2)
        >>> X = np.asarray(X, dtype='float64')
        >>> f_x = f(X)
        >>> M = sorted(range(len(f_x)), key=lambda k: f_x[k], reverse=True)
        >>> M = np.asarray([M], dtype='int8')
        >>> gpr_opt = GProOptimization(X, M, GP_params)
        >>> function_opt = gpr_opt.function_optimization(f=f, bounds=bounds, max_iter=d*10,
        ...                                              warm_up=100, n_iter=1)

        >>> optimal_values, X_post, M_post, f_post = function_opt
        >>> print('optimal values: ', optimal_values)

        >>> # rmse
        >>> print('rmse: ', .5 * sum(np.sqrt((optimal_values - theta['mu'][0])**2)))
        >>> # 2d plot
        >>> if d == 2:
        ...     resolution = 10
        ...     x_min, x_max = bounds['x0'][0], bounds['x0'][1]
        ...     y_min, y_max = bounds['x1'][0], bounds['x1'][1]
        ...     x = np.linspace(x_min, x_max, resolution)
        ...     y = np.linspace(y_min, y_max, resolution)
        ...     X, Y = np.meshgrid(x, y)
        ...     grid = np.empty((resolution ** 2, 2))
        ...     grid[:, 0] = X.flat
        ...     grid[:, 1] = Y.flat
        ...     Z = gm(grid, theta)
        ...     plt.imshow(Z.reshape(-1, resolution), interpolation="bicubic",
        ...                origin="lower", cmap=cm.rainbow,
        ...                extent=[x_min, x_max, y_min, y_max])
        ...     plt.scatter(optimal_values[0], optimal_values[1], color='black', s=10)
        ...     plt.show()

        """

        X, M = check_x_m(self.X, self.M)
        new_pair = M[M.shape[0] - 1]
        for M_ind_cpt in range((M.shape[0] - 1), max_iter + (M.shape[0] - 1)):
            self.fit(X, M, f_prior)
            x_optim = self.bayesopt(bounds, method, warm_up, n_iter)
            f_optim = self.predict(x_optim)
            f_prior = np.concatenate((self.posterior, f_optim))
            X = np.concatenate((X, x_optim))
            # current preference index in X.
            M_ind_current = M[M.shape[0] - 1][0]
            # suggestion index in X.
            M_ind_proposal = M_ind_cpt + 2
            new_pair = np.array([M_ind_current, M_ind_proposal])
            proposal = X[M_ind_proposal].reshape(1, -1)
            current = X[M_ind_current].reshape(1, -1)
            # minimize by convention.
            if f(current) < f(proposal):
                new_pair = np.array([M_ind_proposal, M_ind_current])
            M = np.vstack((M, new_pair))
        optimal_values = X[new_pair[0]]
        f_posterior = f_prior
        return optimal_values, X, M, f_posterior


