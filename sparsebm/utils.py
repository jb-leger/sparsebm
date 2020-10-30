import numpy as np
import scipy
from . import LBM_bernouilli, SBM_bernouilli
from typing import Any, Tuple, Union, Optional
from scipy.sparse import coo_matrix


def lbm_merge_group(
    model: LBM_bernouilli,
    type: int,
    idx_group_1: int,
    idx_group_2: int,
    indices_ones: np.ndarray,
) -> Tuple[float, LBM_bernouilli]:
    """ Given a LBM model, returns the model obtained from the merge of the specified classes.

    Parameters
    ----------
    model : sparsebm.LBM_bernouilli
        The model from which the merge is realized.
    idx_group_1 : int
        index of the first row/column class to merge.
    idx_group_2 : int
        index of the second row/column class to merge.
    type : int
        0 for rows merging, 1 for columns merging.
    indices_ones : numpy.ndarray
        Indices of elements that are non-zero in the original data matrix.
    Returns
    -------
    tuple of (float, sparsebm.LBM_bernouilli)
        The ICL value and the model obtained from the merge of two classes.
    """
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


def sbm_merge_group(
    model: SBM_bernouilli,
    idx_group_1: int,
    idx_group_2: int,
    indices_ones: np.ndarray,
) -> Tuple[float, SBM_bernouilli]:
    """ Given a SBM model, returns the model obtained from the merge of the specified classes.

    Parameters
    ----------
    model : sparsebm.SBM_bernouilli
        The model from which the merge is realized.
    idx_group_1 : int
        index of the first row/column class to merge.
    idx_group_2 : int
        index of the second row/column class to merge.
    indices_ones : numpy.ndarray
        Indices of elements that are non-zero in the original data matrix.
    Returns
    -------
    tuple of (float, sparsebm.SBM_bernouilli)
        The ICL value and the model obtained from the merge of two classes.
    """
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


def lbm_split_group(
    model: LBM_bernouilli,
    row_col_degrees: Tuple[np.ndarray],
    type: int,
    index: int,
    indices_ones: np.array,
) -> Tuple[float, LBM_bernouilli]:
    """ Given a LBM model, returns the model obtained from the split of the specified class.

    The specified class is splitted according to its median of degree.

    Parameters
    ----------
    model : sparsebm.LBM_bernouilli
        The model from which the merge is realized.

    row_col_degrees: tuple of numpy.ndarray
        Tuple of two arrays that contains the row and column degrees of the original data matrix
    type : int
        0 for rows splitting, 1 for columns splitting.
    index : int
        index of the row/column class to split.
    indices_ones : numpy.ndarray
        Indices of elements that are non-zero in the original data matrix.
    Returns
    -------
    tuple of (float, sparsebm.LBM_bernouilli)
        The ICL value and the model obtained from the split of the specified class.
    """
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


def sbm_split_group(
    model: SBM_bernouilli,
    degrees: np.ndarray,
    index: int,
    indices_ones: np.array,
):
    """ Given a SBM model, returns the model obtained from the split of the specified class.

    The specified class is splitted according to its median of degree.

    Parameters
    ----------
    model : sparsebm.SBM_bernouilli
        The model from which the merge is realized.
    degrees: numpy.ndarray
        Array that contains the degrees of the original data matrix.
    index : int
        index of the class to split.
    indices_ones : numpy.ndarray
        Indices of elements that are non-zero in the original data matrix.
    Returns
    -------
    tuple of (float, sparsebm.SBM_bernouilli)
        The ICL value and the model obtained from the split of the specified class.
    """
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


def reorder_rows(X: coo_matrix, idx: np.ndarray) -> None:
    """ Reorders the rows of the COO sparse matrix given in argument.

    Parameters
    ----------
    X : scipy.sparse.coo_matrix
        The sparse matrix to reorder.
    idx: numpy.ndarray,  shape=(X.shape[0],)
        Row indices used to reorder the matrix.
    """
    idx = idx.flatten()
    assert isinstance(
        X, scipy.sparse.coo_matrix
    ), "X must be scipy.sparse.coo_matrix"
    assert X.shape[0] == idx.shape[0], "idx shape[0] must be X shape[0]"
    idx = np.argsort(idx)
    idx = np.asarray(idx, dtype=X.row.dtype)
    X.row = idx[X.row]
