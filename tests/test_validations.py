import numpy as np
import pytest

from GPro.validations import check_kernel


def test_check_kernel():
    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    assert check_kernel(X, length_scale=1.0) is None
    assert check_kernel(X, length_scale=np.array([0.3, 0.4, 0.5])) is None

    with pytest.raises(AssertionError):
        check_kernel(X, length_scale=np.array([0.3, 0.4]))
    with pytest.raises(ValueError):
        check_kernel(X, length_scale="1.0")
    with pytest.raises(TypeError):
        check_kernel(X, length_scale=np.array(["0.3", "0.4", "0.5"]))
    with pytest.raises(ValueError):
        check_kernel(X, length_scale=np.array([0.3, -0.4, 0.5]))
    with pytest.raises(ValueError):
        check_kernel(X, length_scale=np.array([0.3, 0.4, 0.0]))

    # Test nu parameter for Matern kernel:
    assert check_kernel(X, length_scale=1.0, nu=1.5) is None
    with pytest.raises(ValueError):
        check_kernel(X, length_scale=1.0, nu=0.0)
    with pytest.raises(ValueError):
        check_kernel(X, length_scale=1.0, nu=-1.5)
    with pytest.raises(ValueError):
        check_kernel(X, length_scale=1.0, nu="1.5")
