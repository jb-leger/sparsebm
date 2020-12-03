from .. import generate_SBM_dataset

import numpy as np
import itertools


def test_generate_sbm():
    np.random.seed(0)
    n = 10 ** 3
    nq = 4
    alpha = np.ones(nq) / nq
    degree_wanted = 20
    pi_sim = np.array(
        [
            [0.04923077, 0.01846154, 0.00615385, 0.03076923],
            [0.01846154, 0.03692308, 0.0, 0.0],
            [0.00615385, 0.0, 0.05538462, 0.01230769],
            [0.03076923, 0.0, 0.01230769, 0.04307692],
        ]
    )

    data = generate_SBM_dataset(
        n, nq, pi_sim, alpha, symmetric=True, verbosity=False
    )
    X, Y1, = (data["data"], data["cluster_indicator"])
    assert np.abs(pi_sim.mean() - X.nnz / np.prod(X.shape)) < 10 ** -3
    assert X.shape == (n, n)
    assert np.all(Y1[0] == np.array([0, 1, 0, 0]))
    assert Y1.shape == (n, nq)
