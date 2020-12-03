from .. import SBM, generate_SBM_dataset

import numpy as np
import time
import itertools


def test_sbm():
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

    data = generate_SBM_dataset(n, nq, pi_sim, alpha, symmetric=True)
    X, Y1, = (data["data"], data["cluster_indicator"])

    model = SBM(
        nq,
        max_iter=10,
        n_init=1,
        n_init_total_run=1,
        n_iter_early_stop=1,
        atol=1e-5,
        verbosity=0,
        use_gpu=False,
    )
    model.fit(X, symmetric=True)

    pi = model.pi_
    bp = max(
        itertools.permutations(range(nq)),
        key=lambda permut: (model.tau_[:, permut] * Y1).sum(),
    )

    assert np.max(np.abs(pi[:, bp][bp, :] - pi_sim)) < 0.04
