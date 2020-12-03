from .. import LBM, generate_LBM_dataset

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

    data = generate_LBM_dataset(n1, n2, nq, nl, pi_sim, alpha_1, alpha_2)
    X, Y1, Y2 = (
        data["data"],
        data["row_cluster_indicator"],
        data["column_cluster_indicator"],
    )

    model = LBM(
        nq,
        nl,
        max_iter=10,
        n_init=1,
        n_init_total_run=1,
        n_iter_early_stop=1,
        atol=1e-5,
        verbosity=1,
        use_gpu=False,
    )
    model.fit(X)
    pi = model.pi_

    permutations_lines = list(itertools.permutations(range(nq)))
    permutations_col = list(itertools.permutations(range(nl)))
    res = []
    for permut_col in permutations_col:
        pi_2 = pi[:, permut_col]
        for permut_ligne in permutations_lines:
            pi_3 = pi_2[permut_ligne, :]
            res.append(
                (np.max(np.abs(pi_sim - pi_3)), permut_ligne, permut_col)
            )
    res.sort()

    assert res[0][0] < 0.07
