from scipy.stats import norm
from inspect import signature


class Acquisition:
    """Base class for all acquisition functions.

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
            raise RuntimeError("GPro acquisition methods should"
                               " always specify their parameters in the"
                               " signature of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls,))
        for arg in args:
            params[arg] = getattr(self, arg, None)
        return params

    def set_params(self, **params):
        """Set the parameters of an acquisition function.

        The method works on simple acquisition functions as well as on
        nested acquisition functions. The latter have parameters of the
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
                    raise ValueError('Invalid parameter %s for acquisition '
                                     'function %s. '
                                     'Check the list of available parameters '
                                     'with `Acquisition.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params({sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for acquisition '
                                     'function %s. '
                                     'Check the list of available parameters '
                                     'with `Acquisition.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self


class ExpectedImprovement(Acquisition):
    """Expected improvement utility function.

     Parameters
     ----------
     xi : float, default: 0
     Exploitation-exploration trade-off parameter.

    """

    def __init__(self, xi=0):
        self.xi = xi

    def __call__(self, y_mean, std, y_max):
        z = (y_mean - y_max - self.xi) / std
        res = (y_mean - y_max - self.xi) * norm.cdf(z) + std * norm.pdf(z)
        return res


class UCB(Acquisition):
    """Upper confidence bound utility function.

    Parameters
    ----------
    kappa : float, default: 0
    Exploitation-exploration trade-off parameter.

    """

    def __init__(self, kappa=1):
        self.kappa = kappa

    def __call__(self, y_mean, std, y_max):
        return y_mean + self.kappa * std
