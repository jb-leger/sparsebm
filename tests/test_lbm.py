from .. import LBM_bernouilli, generate_bernouilli_LBM

import numpy as np
import time
import itertools
from scipy import optimize
import itertools


def test_lbm():
    np.random.seed(0)

    F = 10 ** 3
    n1, n2 = F, int(1.5 * F)
    nq, nl = 3, 4
    alpha_1 = np.ones(nq) / nq
    alpha_2 = np.ones(nl) / nl
    pi_sim = np.array([[8, 1, 1, 4], [1, 8, 1, 4], [0, 1, 8, 0]]) / (0.08 * F)

    data = generate_bernouilli_LBM(n1, n2, nq, nl, pi_sim, alpha_1, alpha_2)
    X, Y1, Y2 = (data["X"], data["Y1"], data["Y2"])

    model = LBM_bernouilli(
        nq,
        nl,
        max_iter=10000,
        n_init=25,
        n_init_total_run=10,
        n_iter_early_stop=100,
        tol=1e-5,
        verbosity=1,
        gpu_number=None,
    )
    model.fit(X)

    pi = model._pi

    permutations_lines = list(itertools.permutations(range(nq)))
    permutations_col = list(itertools.permutations(range(nl)))
    res = []
    for permut_col in permutations_col:
        pi_2 = pi[:, permut_col]
        for permut_ligne in permutations_lines:
            pi_3 = pi_2[permut_ligne, :]
            res.append((np.max(np.abs(pi_sim - pi_3)), permut_ligne, permut_col))
    res.sort()

    assert res[0][0] < 0.002
