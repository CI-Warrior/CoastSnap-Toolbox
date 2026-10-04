import os

import pytest
from scipy.io import loadmat

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="session")
def ref():
    """Outputs of the original MATLAB code (see tests/matlab_reference/make_reference.m)."""
    return loadmat(os.path.join(DATA, "matlab_reference.mat"), simplify_cells=True)
