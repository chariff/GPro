from inspect import signature
from scipy.stats import norm
import numpy as np
from itertools import compress


class PosteriorApproximation:
    """Base class for all posterior approximation strategies.

    .. versionadded:: 0.1
    """

    def get_params(self):
        """Get the parameters of an acquisition function.

        Returns
        -------
        params : mapping of string to any parameter names
        mapped to their values.
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
            raise RuntimeError("GPro posterior approximation methods should"
                               " always specify their parameters in the"
                               " signature of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls,))
        for arg in args:
            params[arg] = getattr(self, arg, None)
        return params

    def set_params(self, **params):
        """Set the parameters of this posterior approximation.

        The method works on simple posterior approximation as well as on
        nested posterior approximation. The latter have parameters of the
        form ``<component>__<parameter>`` so that it's possible to
        update each component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            return self
        valid_params = self.get_params()
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for posterior '
                                     'approximation %s. '
                                     'Check the list of available parameters '
                                     'with `kernel.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params({sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for posterior '
                                     'approximation %s. '
                                     'Check the list of available parameters '
                                     'with `kernel.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self


class Laplace(PosteriorApproximation):
    """Laplace approximation of the posterior distribution P(f|M) in
    a probit-preference gaussian regression model.

    Parameters
    ----------
    s_eval : float, optional, default: 1e-5
            Noise of the latent preference relations functions.

    max_iter : int , optional, default: 1000
            Maximum number of iterations of the Newton-Raphson
            recursion for the Laplace approximation of the
            posterior distribution P(f|M).

    eta : float, optional, default: 0.01
        Gradient descent step size.

    tol : float, optional, default: 1e-5
        Gradient descent convergence tolerance.
    """
    def __init__(self, s_eval=1e-5, max_iter=1000, eta=0.01, tol=1e-5):
        self.s_eval = s_eval
        self.max_iter = max_iter
        self.eta = eta
        self.tol = tol

    def __call__(self, f, M, K):
        """Return the Laplace approximation of P(f|M).
        A Newton-Raphson descent is used to approximate f
        at the MAP.

        Parameters
        ----------
        f : Gaussian process prior.

        M : array-like, shape = (n_samples - 1, 2)
            Target choices. A preference is an array of positive
            integers of shape = (2,). preference[0], r, is an index
            of X preferred over preference[1], c, which is an
            index of X.

        K : array.
            Kernel(X, X)

        Returns
        -------
        f_new  : array-like, shape = (n_samples, 1)
        Laplace approximation of P(f|M).

        """

        def z(f, M):
            """Likelihood function of a preference relation."""
            r, c = M[:, 0], M[:, 1]
            return ((f[r] - f[c]) / np.sqrt(2) * self.s_eval).flatten()

        def delta(f, M, K):
            """Root of the Taylor expansion derivative of log P(f|M)
            with respect to the latent preference valuation functions."""

            # Quantities of the first order derivative
            # of the loss function p(f|M) with respect to f.
            n = len(f)
            b = np.zeros(n)
            m_set = np.linspace(0, n - 1, n, dtype=int)
            for i in m_set:
                t_r, t_c = M[:, 0] == i, M[:, 1] == i
                ind_r = list(compress(range(len(t_r)), t_r))
                ind_c = list(compress(range(len(t_c)), t_c))
                # Likelihood function of a preference relation.
                z_r = z(f, M[ind_r, :])
                z_c = z(f, M[ind_c, :])
                pos_r = norm.pdf(z_r) / norm.cdf(z_r)
                neg_c = norm.pdf(z_c) / norm.cdf(z_c)
                b[i] = (sum(pos_r) - sum(neg_c)) / (np.sqrt(2))
            # Quantities of the second order derivative
            # of the loss function p(f|M) with respect to f.
            # c can be shown to be positive semi-definite.
            c = np.zeros((n, n))
            M_uni = np.unique(M, axis=0)
            for i in range(M_uni.shape[0]):
                m, n = M_uni[i, 0], M_uni[i, 1]
                z_mn = z(f, M_uni[[i], :])
                z_nm = -z_mn
                pdf_z = norm.pdf(z_mn)
                cdf_z_mn = norm.cdf(z_mn)
                cdf_z_nm = norm.cdf(z_nm)
                c_mn = (pdf_z / cdf_z_mn) ** 2 + pdf_z / cdf_z_mn * z_mn
                c_nm = (pdf_z / cdf_z_nm) ** 2 + pdf_z / cdf_z_nm * z_nm
                c[m][n] = -(c_mn + c_nm) / 2 * self.s_eval
                c[n][m] = -(c_mn + c_nm) / 2 * self.s_eval
            # Gradient
            Kf = np.linalg.solve(K, f)
            g = Kf.flatten() - b
            # Hessian
            H = np.linalg.inv(K) + c
            return g, H

        # Newton-Raphson descent
        f_new = np.empty(shape=(M.shape[0] + 1, 1))
        f_old = f
        eps = self.tol + 1
        iteration = 0
        while iteration < self.max_iter and eps > self.tol:
            g, H = delta(f_old, M, K)
            f_new = f_old - (self.eta * np.linalg.solve(H, g).reshape(-1, 1))
            eps = np.linalg.norm(f_new - f_old, ord=2)
            f_old = f_new
            iteration += 1
        return f_new
