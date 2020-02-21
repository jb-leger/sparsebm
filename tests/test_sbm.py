from .. import SBM_bernouilli, generate_bernouilli_SBM

import numpy as np
import time
import itertools
from scipy import optimize


def test_sbm():
    np.random.seed(0)
    n = 10 ** 3
    nq = 4
    alpha = np.ones(nq) / nq
    degree_wanted = 20
    pi_sim = np.array([[8, 3, 1, 5], [3, 6, 0, 0], [1, 0, 9, 2], [5, 0, 2, 7]])

    def f(x, n, pi, degree_wanted):
        return np.abs((pi / (x * n)).mean() * n - degree_wanted)

    res = optimize.minimize_scalar(lambda x: f(x, n, pi_sim, degree_wanted))
    pi_sim = pi_sim / (res.x * n)
    degree_moyen = pi_sim.mean() * n

    data = generate_bernouilli_SBM(n, nq, pi_sim, alpha, symetric=True)
    X, Y1, = (data["X"], data["Y"])

    model = SBM_bernouilli(
        nq,
        max_iter=10000,
        n_init=25,
        n_init_total_run=10,
        n_iter_early_stop=100,
        tol=1e-5,
        verbosity=0,
        gpu_number=None,
        symetric=True,
    )
    model.fit(X)

    pi = model._pi
    bp = max(
        itertools.permutations(range(nq)),
        key=lambda permut: (model._tau[:, permut] * Y1).sum(),
    )

    assert np.max(np.abs(pi[:, bp][bp, :] - pi_sim)) < 0.002
