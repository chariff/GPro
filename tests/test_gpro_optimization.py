import pytest
import mock
from GPro.optimization import GProOptimization
from GPro.kernels import Matern
from GPro.posterior import Laplace
from GPro.acquisitions import UCB
import numpy as np


def f(x):
    return x.sum()


bounds = {'x1': (0, 10), 'x2': (0, 10)}

X = np.array([[1, 0], [2, 5]]).reshape(-1, 2)
M = np.array([1, 0]).reshape(-1, 2)


def test_optimization():
    gpr_opt = GProOptimization(X, M)
    function_opt = gpr_opt.function_optimization(f=f, bounds=bounds,
                                                 max_iter=1,
                                                 n_init=1,
                                                 n_solve=1)

    optimal_values, X_post, M_post, f_post = function_opt
    assert len(optimal_values) == 2
    assert len(X_post) == 3
    assert len(M_post) == 2
    assert len(f_post) == 3

    # Use posterior as prior
    gpr_opt = GProOptimization(X_post, M_post)
    function_opt = gpr_opt.function_optimization(f=f, bounds=bounds,
                                                 max_iter=1,
                                                 n_init=1,
                                                 n_solve=1,
                                                 f_prior=f_post)
    optimal_values, X_post, M_post, f_post = function_opt
    assert len(optimal_values) == 2
    assert len(X_post) == 4
    assert len(M_post) == 3
    assert len(f_post) == 4


def test_params():
    # default GP_parameters
    gpr_opt = GProOptimization(X, M)
    function_opt = gpr_opt.function_optimization(f=f, bounds=bounds,
                                                 max_iter=1,
                                                 n_init=1,
                                                 n_solve=1)

    optimal_values, X_post, M_post, f_post = function_opt
    assert len(optimal_values) == 2
    assert len(X_post) == 3
    assert len(M_post) == 2
    assert len(f_post) == 3

    kernel_params = gpr_opt.kernel.get_params()
    assert len(kernel_params) == 1
    assert 'length_scale' in kernel_params
    assert not gpr_opt.kernel.anisotropic
    gpr_opt.kernel.set_params(**{'length_scale': 2})
    kernel_params = gpr_opt.kernel.get_params()
    assert kernel_params['length_scale'] == 2

    acquisition_params = gpr_opt.acquisition.get_params()
    assert len(acquisition_params) == 1
    assert 'xi' in acquisition_params
    gpr_opt.acquisition.set_params(**{'xi': 2})
    acquisition_params = gpr_opt.acquisition.get_params()
    assert acquisition_params['xi'] == 2

    post_approx_params = gpr_opt.post_approx.get_params()
    assert len(post_approx_params) == 4
    assert all([key in post_approx_params for key in
                ['s_eval', 'max_iter', 'eta', 'tol']])
    gpr_opt.post_approx.set_params(**{'s_eval': 1})
    post_approx_params = gpr_opt.post_approx.get_params()
    assert post_approx_params['s_eval'] == 1

    # custom GP_parameters
    GP_params = {'kernel': Matern(length_scale=[1, 1], nu=2.5),
                 'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
                                        eta=0.01, tol=1e-3),
                 'acquisition': UCB(kappa=2.576),
                 'alpha': [1e-5],
                 'random_state': 0}

    gpr_opt = GProOptimization(X, M, GP_params)
    function_opt = gpr_opt.function_optimization(f=f, bounds=bounds,
                                                 max_iter=1,
                                                 n_init=1,
                                                 n_solve=1)

    optimal_values, X_post, M_post, f_post = function_opt
    assert len(optimal_values) == 2
    assert len(X_post) == 3
    assert len(M_post) == 2
    assert len(f_post) == 3

    kernel_params = gpr_opt.kernel.get_params()
    assert len(kernel_params) == 2
    assert 'nu' in kernel_params
    gpr_opt.kernel.set_params(**{'nu': 1.5})
    assert gpr_opt.kernel.anisotropic
    kernel_params = gpr_opt.kernel.get_params()
    assert kernel_params['nu'] == 1.5

    acquisition_params = gpr_opt.acquisition.get_params()
    assert len(acquisition_params) == 1
    assert 'kappa' in acquisition_params
    gpr_opt.acquisition.set_params(**{'kappa': 2})
    acquisition_params = gpr_opt.acquisition.get_params()
    assert acquisition_params['kappa'] == 2

    post_approx_params = gpr_opt.post_approx.get_params()
    assert len(post_approx_params) == 4
    assert all([key in post_approx_params for key in
                ['s_eval', 'max_iter', 'eta', 'tol']])
    gpr_opt.post_approx.set_params(**{'s_eval': 1})
    post_approx_params = gpr_opt.post_approx.get_params()
    assert post_approx_params['s_eval'] == 1

    # test console optimization

    with mock.patch('builtins.input', return_value="Q"):
        assert len(gpr_opt.console_optimization(bounds=bounds, n_solve=1)) == 4

    with mock.patch('builtins.input', return_value=""):
        assert len(gpr_opt.console_optimization(bounds=bounds, n_solve=1)) == 4

    with mock.patch('builtins.input', return_value="p"):
        assert len(gpr_opt.console_optimization(bounds=bounds,
                                                n_solve=1, max_iter=1)) == 4

    with mock.patch('builtins.input', return_value="s"):
        assert len(gpr_opt.console_optimization(bounds=bounds,
                                                n_solve=1, max_iter=1)) == 4

    gpr_opt.kernel.set_params(**{'nu': .5, 'length_scale': 1})
    with mock.patch('builtins.input', return_value="s"):
        assert len(gpr_opt.console_optimization(bounds=bounds,
                                                n_solve=1, max_iter=1)) == 4
    gpr_opt.kernel.set_params(**{'nu': 3})
    with mock.patch('builtins.input', return_value="s"):
        assert len(gpr_opt.console_optimization(bounds=bounds,
                                                n_solve=1, max_iter=1)) == 4

    gpr_opt.acquisition.set_params()
    gpr_opt.kernel.set_params()
    gpr_opt.post_approx.set_params()


if __name__ == '__main__':
    pytest.main([__file__])
