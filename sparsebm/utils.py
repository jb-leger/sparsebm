import numpy as np


def lbm_merge_group(model, type, idx_group_1, idx_group_2, indices_ones):
    if type != 0 and type != 1:
        print("Type error in merge group")
        assert False
    eps = 1e-4
    if type == 0:
        model._n_row_clusters -= 1
        t = model._tau_1
        alpha = model._alpha_1
    else:
        model._n_column_clusters -= 1
        t = model._tau_2
        alpha = model._alpha_2
    if idx_group_1 > idx_group_2:
        c = idx_group_2
        idx_group_2 = idx_group_1
        idx_group_1 = c
    n = t.shape[0]

    new_t = np.delete(t, idx_group_2, axis=1)
    new_t[:, idx_group_1] = t[:, idx_group_1] + t[:, idx_group_2]
    new_alpha = np.delete(alpha, idx_group_2)
    new_alpha[idx_group_1] = alpha[idx_group_1] + alpha[idx_group_2]

    new_pi = np.delete(model._pi, idx_group_2, axis=type)

    if type == 0:
        model._tau_1 = new_t
        model._alpha_1 = new_alpha
        new_pi[idx_group_1] = (
            alpha[idx_group_1] * model._pi[idx_group_1]
            + alpha[idx_group_2] * model._pi[idx_group_2]
        ) / (alpha[idx_group_1] + alpha[idx_group_2])
    else:
        model._tau_2 = new_t
        model._alpha_2 = new_alpha
        new_pi[:, idx_group_1] = (
            alpha[idx_group_1] * model._pi[:, idx_group_1]
            + alpha[idx_group_2] * model._pi[:, idx_group_2]
        ) / (alpha[idx_group_1] + alpha[idx_group_2])

    model._pi = new_pi
    nq = model._n_row_clusters
    nl = model._n_column_clusters

    # Transfert to GPU if necessary
    t1 = model._np.asarray(model._tau_1)
    t2 = model._np.asarray(model._tau_2)
    a1 = model._np.asarray(model._alpha_1)
    a2 = model._np.asarray(model._alpha_2)
    pi = model._np.asarray(model._pi)
    ll = model._compute_likelihood(indices_ones, pi, a1, a2, t1, t2)
    model._loglikelihood = ll if model.use_gpu else ll
    return (model.get_ICL(), model)


def sbm_merge_group(model, idx_group_1, idx_group_2, indices_ones):
    eps = 1e-4
    model._n_clusters -= 1
    t = model._tau
    alpha = model._alpha

    if idx_group_1 > idx_group_2:
        c = idx_group_2
        idx_group_2 = idx_group_1
        idx_group_1 = c
    n = t.shape[0]

    new_t = np.delete(t, idx_group_2, axis=1)
    new_t[:, idx_group_1] = t[:, idx_group_1] + t[:, idx_group_2]
    new_alpha = np.delete(alpha, idx_group_2)
    new_alpha[idx_group_1] = alpha[idx_group_1] + alpha[idx_group_2]

    model._alpha = new_alpha
    model._tau = new_t
    nq = model._n_clusters

    # Transfert to GPU if necessary
    t1 = model._np.asarray(model._tau)
    a1 = model._np.asarray(model._alpha)
    t1_sum = t1.sum(0)
    pi = (
        t1[indices_ones[0]].reshape(-1, nq, 1)
        * t1[indices_ones[1]].reshape(-1, 1, nq)
    ).sum(0) / ((t1_sum.reshape((-1, 1)) * t1_sum) - t1.T @ t1)

    ll = model._compute_likelihood(indices_ones, pi, a1, t1)
    model._pi = pi if model.use_gpu else pi
    model._loglikelihood = ll if model.use_gpu else ll
    return (model.get_ICL(), model)


def lbm_split_group(model, row_col_degrees, type, index, indices_ones):
    if type != 0 and type != 1:
        print("Type error in split group")
        assert False
    eps = 1e-4
    if type == 0:
        model._n_row_clusters += 1
        t = model._tau_1
    else:
        model._n_column_clusters += 1
        t = model._tau_2
    n = t.shape[0]
    degrees = row_col_degrees[type].flatten()
    mask = t.argmax(1) == index
    if not np.any(mask):
        return (-np.inf, model)
    median = np.median(degrees[mask])
    t = np.concatenate((t, eps * np.ones((n, 1))), 1)
    t[(degrees > median) & mask, index] -= eps
    t[(degrees <= median) & mask, -1] = t[(degrees <= median) & mask, index]
    t[(degrees <= median) & mask, index] = eps
    t /= t.sum(1).reshape(-1, 1)

    if type == 0:
        model._tau_1 = t
        model._alpha_1 = t.mean(0)
    else:
        model._tau_2 = t
        model._alpha_2 = t.mean(0)

    nq = model._n_row_clusters
    nl = model._n_column_clusters

    # Transfert to GPU if necessary
    t1 = model._np.asarray(model._tau_1)
    t2 = model._np.asarray(model._tau_2)
    a1 = model._np.asarray(model._alpha_1)
    a2 = model._np.asarray(model._alpha_2)

    pi = (
        t1[indices_ones[0]].reshape(-1, nq, 1)
        * t2[indices_ones[1]].reshape(-1, 1, nl)
    ).sum(0) / (t1.sum(0).reshape(nq, 1) * t2.sum(0).reshape(1, nl))

    model._pi = pi.get() if model.use_gpu else pi
    ll = model._compute_likelihood(indices_ones, pi, a1, a2, t1, t2)
    model._loglikelihood = ll if model.use_gpu else ll

    return (model.get_ICL(), model)


def sbm_split_group(model, degrees, index, indices_ones):
    eps = 1e-4
    model._n_clusters += 1
    t = model._tau
    n = t.shape[0]
    degrees = degrees.flatten()
    mask = t.argmax(1) == index
    if not np.any(mask):
        return (-np.inf, model)
    median = np.median(degrees[mask])
    t = np.concatenate((t, eps * np.ones((n, 1))), 1)
    t[(degrees > median) & mask, index] -= eps
    t[(degrees <= median) & mask, -1] = t[(degrees <= median) & mask, index]
    t[(degrees <= median) & mask, index] = eps
    t /= t.sum(1).reshape(-1, 1)

    model._tau = t
    model._alpha = t.mean(0)
    nq = model._n_clusters

    # Transfert to GPU if necessary
    t1 = model._np.asarray(model._tau)
    a1 = model._np.asarray(model._alpha)
    t1_sum = t1.sum(0)

    pi = (
        t1[indices_ones[0]].reshape(-1, nq, 1)
        * t1[indices_ones[1]].reshape(-1, 1, nq)
    ).sum(0) / ((t1_sum.reshape((-1, 1)) * t1_sum) - t1.T @ t1)

    model._pi = pi.get() if model.use_gpu else pi
    ll = model._compute_likelihood(indices_ones, pi, a1, t1)
    model._loglikelihood = ll if model.use_gpu else ll

    return (model.get_ICL(), model)
