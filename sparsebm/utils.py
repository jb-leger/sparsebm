import numpy as np
import scipy
from . import LBM, SBM
from typing import Tuple, Union, Optional
from scipy.sparse import coo_matrix
from scipy.special import comb
import logging

logger = logging.getLogger(__name__)


def ARI(
    labels_true: Union[np.ndarray, list], labels_pred: Union[np.ndarray, list]
) -> float:
    """
    Adjusted Rand Index.

    The Adjusted Rand Index (ARI) computes a similarity measure
    between two clusterings and was developed by Hubert and
    Arabie (1985). This index is symmetric and takes the value 1 when the
    partitions agree perfectly up to a permutation.

    Parameters
    ----------
    labels_true : int array, shape = (n_samples,)
        Ground truth class labels of the partition used as reference

    labels_pred : int array, shape = (n_samples,)
        Cluster labels of the partition to evaluate


    Returns
    -------
    ari : float
       Similarity score between -1.0 and 1.0. Random labelings have a ARI
       close to 0.0. 1.0 stands for perfect match.

    Examples
    --------
      >>> ARI(
            [0, 0, 1, 2, 2, 2],
            [0, 0, 1, 1, 2, 2]
        )
      0.4444444444444444

    References
    ----------
    .. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075

    """
    labels_true = np.array(labels_true).flatten()
    labels_pred = np.array(labels_pred).flatten()
    assert labels_true.size == labels_pred.size

    n = labels_true.size
    nb_true_class = len(set(labels_true))
    nb_pred_class = len(set(labels_pred))

    if (
        nb_true_class == nb_pred_class == 1
        or nb_true_class == nb_pred_class == 0
        or nb_true_class == nb_pred_class == n
    ):
        return 1.0

    _, true_class_idx = np.unique(labels_true, return_inverse=True)
    _, pred_class_idx = np.unique(labels_pred, return_inverse=True)
    contingency_table = np.zeros((nb_true_class, nb_pred_class))
    np.add.at(contingency_table, (true_class_idx, pred_class_idx), 1)

    sum_tt_comb = comb(contingency_table, 2).sum()
    sum_a_comb = comb(contingency_table.sum(axis=1), 2).sum()
    sum_b_comb = comb(contingency_table.sum(axis=0), 2).sum()
    comb_n = comb(n, 2).sum()

    ari = ((sum_tt_comb) - (sum_a_comb * sum_b_comb / comb_n)) / (
        0.5 * (sum_a_comb + sum_b_comb) - (sum_a_comb * sum_b_comb) / comb_n
    )
    return ari


def CARI(
    labels_true_part_1: Union[np.ndarray, list],
    labels_true_part_2: Union[np.ndarray, list],
    labels_pred_part_1: Union[np.ndarray, list],
    labels_pred_part_2: Union[np.ndarray, list],
) -> float:
    """Coclustering Adjusted Rand Index for two sets of biclusters.

    The Coclustering Adjuster Rand Index (CARI) computes a similarity measure
    between two coclusterings and is an adaptation of the
    Adjusted Rand Index (ARI) developed by Hubert and Arabie (1985) from a
    coclustering point of view.
    Like the ARI, this index is symmetric and takes the value 1 when the
    couples of partitions agree perfectly up to a permutation.

    Parameters
    ----------
    labels_true_part_1 : int array, shape = (n_samples_1,)
        Ground truth class labels of the first partition used as reference

    labels_true_part_2 : int array, shape = (n_samples_2,)
        Ground truth class labels of the second partition used as reference

    labels_pred_part_1 : int array, shape = (n_samples_1,)
        Cluster labels of the fist partition to evaluate

    labels_pred_part_2 : int array, shape = (n_samples_2,)
        Cluster labels of the second partition to evaluate

    Returns
    -------
    cari : float
       Similarity score between -1.0 and 1.0. Random labelings have a CARI
       close to 0.0. 1.0 stands for perfect match.

    Examples
    --------
      >>> CARI(
            [0, 0, 1, 1],
            [0, 0, 1, 2, 2, 2],
            [0, 0, 1, 1],
            [0, 0, 1, 1, 2, 2]
        )
      0.649746192893401

    References
    ----------
    .. [Robert2019] Valérie Robert, Yann Vasseur, Vincent Brault.
      Comparing high dimensional partitions with the Coclustering Adjusted Rand
      Index. 2019. https://hal.inria.fr/hal-01524832v4

    .. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075

    """

    labels_true_part_1 = np.array(labels_true_part_1).flatten()
    labels_true_part_2 = np.array(labels_true_part_2).flatten()
    labels_pred_part_1 = np.array(labels_pred_part_1).flatten()
    labels_pred_part_2 = np.array(labels_pred_part_2).flatten()

    assert labels_true_part_1.size == labels_pred_part_1.size
    assert labels_true_part_2.size == labels_pred_part_2.size

    n_samples_part_1 = labels_true_part_1.size
    n_samples_part_2 = labels_true_part_2.size

    n_classes_part_1 = len(set(labels_true_part_1))
    n_clusters_part_1 = len(set(labels_pred_part_1))
    n_classes_part_2 = len(set(labels_true_part_2))
    n_clusters_part_2 = len(set(labels_pred_part_2))

    if (
        (
            n_classes_part_1
            == n_clusters_part_1
            == n_classes_part_2
            == n_clusters_part_2
            == 1
        )
        or n_classes_part_1
        == n_clusters_part_1
        == n_classes_part_2
        == n_clusters_part_2
        == 0
        or (
            n_classes_part_1 == n_clusters_part_1 == n_samples_part_1
            and n_classes_part_2 == n_clusters_part_2 == n_samples_part_2
        )
    ):
        return 1.0

    # Compute the contingency data tables
    _, true_class_idx_part_1 = np.unique(
        labels_true_part_1, return_inverse=True
    )
    _, pred_class_idx_part_1 = np.unique(
        labels_pred_part_1, return_inverse=True
    )
    contingency_part_1 = np.zeros((n_classes_part_1, n_clusters_part_1))
    np.add.at(
        contingency_part_1, (true_class_idx_part_1, pred_class_idx_part_1), 1
    )
    _, true_class_idx_part_2 = np.unique(
        labels_true_part_2, return_inverse=True
    )
    _, pred_class_idx_part_2 = np.unique(
        labels_pred_part_2, return_inverse=True
    )
    contingency_part_2 = np.zeros((n_classes_part_2, n_clusters_part_2))
    np.add.at(
        contingency_part_2, (true_class_idx_part_2, pred_class_idx_part_2), 1
    )

    # Theorem 3.3 of Robert2019 (https://hal.inria.fr/hal-01524832v4) defines
    # the final contingency matrix by the Kronecker product between the two
    # contingency matrices of patition 1 and 2.
    contingency_table = np.kron(contingency_part_1, contingency_part_2)
    sum_tt_comb = comb(contingency_table, 2).sum()
    sum_a_comb = comb(contingency_table.sum(axis=1), 2).sum()
    sum_b_comb = comb(contingency_table.sum(axis=0), 2).sum()
    comb_n = comb(n_samples_part_1 * n_samples_part_2, 2).sum()

    ari = ((sum_tt_comb) - (sum_a_comb * sum_b_comb / comb_n)) / (
        0.5 * (sum_a_comb + sum_b_comb) - (sum_a_comb * sum_b_comb) / comb_n
    )
    return ari


def lbm_merge_group(
    model: LBM,
    type: int,
    idx_group_1: int,
    idx_group_2: int,
    indices_ones: np.ndarray,
) -> Tuple[float, LBM]:
    """ Given a LBM model, returns the model obtained from the merge of the specified classes.

    Parameters
    ----------
    model : sparsebm.LBM
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
    tuple of (float, sparsebm.LBM)
        The ICL value and the model obtained from the merge of two classes.
    """
    if type != 0 and type != 1:
        logger.error("Type error in merge group")
        assert False
    eps = 1e-4
    if type == 0:
        model.n_row_clusters -= 1
        t = model.tau_1_
        alpha = model.alpha_1_
    else:
        model.n_column_clusters -= 1
        t = model.tau_2_
        alpha = model.alpha_2_
    if idx_group_1 > idx_group_2:
        c = idx_group_2
        idx_group_2 = idx_group_1
        idx_group_1 = c
    n = t.shape[0]

    new_t = np.delete(t, idx_group_2, axis=1)
    new_t[:, idx_group_1] = t[:, idx_group_1] + t[:, idx_group_2]
    new_alpha = np.delete(alpha, idx_group_2)
    new_alpha[idx_group_1] = alpha[idx_group_1] + alpha[idx_group_2]

    new_pi = np.delete(model.pi_, idx_group_2, axis=type)

    if type == 0:
        model.tau_1_ = new_t
        model.alpha_1_ = new_alpha
        new_pi[idx_group_1] = (
            alpha[idx_group_1] * model.pi_[idx_group_1]
            + alpha[idx_group_2] * model.pi_[idx_group_2]
        ) / (alpha[idx_group_1] + alpha[idx_group_2])
    else:
        model.tau_2_ = new_t
        model.alpha_2_ = new_alpha
        new_pi[:, idx_group_1] = (
            alpha[idx_group_1] * model.pi_[:, idx_group_1]
            + alpha[idx_group_2] * model.pi_[:, idx_group_2]
        ) / (alpha[idx_group_1] + alpha[idx_group_2])

    model.pi_ = new_pi
    nq = model.n_row_clusters
    nl = model.n_column_clusters

    # Transfert to GPU if necessary
    t1 = model._np.asarray(model.tau_1_)
    t2 = model._np.asarray(model.tau_2_)
    a1 = model._np.asarray(model.alpha_1_)
    a2 = model._np.asarray(model.alpha_2_)
    pi = model._np.asarray(model.pi_)
    ll = model._compute_likelihood(indices_ones, pi, a1, a2, t1, t2)
    model.loglikelihood_ = ll.get() if model.use_gpu else ll
    return (model.get_ICL(), model)


def sbm_merge_group(
    model: SBM, idx_group_1: int, idx_group_2: int, indices_ones: np.ndarray
) -> Tuple[float, SBM]:
    """ Given a SBM model, returns the model obtained from the merge of the specified classes.

    Parameters
    ----------
    model : sparsebm.SBM
        The model from which the merge is realized.
    idx_group_1 : int
        index of the first row/column class to merge.
    idx_group_2 : int
        index of the second row/column class to merge.
    indices_ones : numpy.ndarray
        Indices of elements that are non-zero in the original data matrix.
    Returns
    -------
    tuple of (float, sparsebm.SBM)
        The ICL value and the model obtained from the merge of two classes.
    """
    eps = 1e-4
    model.n_clusters -= 1
    t = model.tau_
    alpha = model.alpha_

    if idx_group_1 > idx_group_2:
        c = idx_group_2
        idx_group_2 = idx_group_1
        idx_group_1 = c
    n = t.shape[0]

    new_t = np.delete(t, idx_group_2, axis=1)
    new_t[:, idx_group_1] = t[:, idx_group_1] + t[:, idx_group_2]
    new_alpha = np.delete(alpha, idx_group_2)
    new_alpha[idx_group_1] = alpha[idx_group_1] + alpha[idx_group_2]

    model.alpha_ = new_alpha
    model.tau_ = new_t
    nq = model.n_clusters

    # Transfert to GPU if necessary
    t1 = model._np.asarray(model.tau_)
    a1 = model._np.asarray(model.alpha_)
    t1_sum = t1.sum(0)
    pi = (
        t1[indices_ones[0]].reshape(-1, nq, 1)
        * t1[indices_ones[1]].reshape(-1, 1, nq)
    ).sum(0) / ((t1_sum.reshape((-1, 1)) * t1_sum) - t1.T @ t1)

    ll = model._compute_likelihood(indices_ones, pi, a1, t1)
    model.pi_ = pi.get() if model.use_gpu else pi
    model.loglikelihood_ = ll.get() if model.use_gpu else ll
    return (model.get_ICL(), model)


def lbm_split_group(
    model: LBM,
    row_col_degrees: Tuple[np.ndarray],
    type: int,
    index: int,
    indices_ones: np.array,
) -> Tuple[float, LBM]:
    """ Given a LBM model, returns the model obtained from the split of the specified class.

    The specified class is splitted according to its median of degree.

    Parameters
    ----------
    model : sparsebm.LBM
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
    tuple of (float, sparsebm.LBM)
        The ICL value and the model obtained from the split of the specified class.
    """
    if type != 0 and type != 1:
        logger.error("Type error in split group")
        assert False
    eps = 1e-4
    if type == 0:
        model.n_row_clusters += 1
        t = model.tau_1_
    else:
        model.n_column_clusters += 1
        t = model.tau_2_
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
        model.tau_1_ = t
        model.alpha_1_ = t.mean(0)
    else:
        model.tau_2_ = t
        model.alpha_2_ = t.mean(0)

    nq = model.n_row_clusters
    nl = model.n_column_clusters

    # Transfert to GPU if necessary
    t1 = model._np.asarray(model.tau_1_)
    t2 = model._np.asarray(model.tau_2_)
    a1 = model._np.asarray(model.alpha_1_)
    a2 = model._np.asarray(model.alpha_2_)

    pi = (
        t1[indices_ones[0]].reshape(-1, nq, 1)
        * t2[indices_ones[1]].reshape(-1, 1, nl)
    ).sum(0) / (t1.sum(0).reshape(nq, 1) * t2.sum(0).reshape(1, nl))

    model.pi_ = pi.get() if model.use_gpu else pi
    ll = model._compute_likelihood(indices_ones, pi, a1, a2, t1, t2)
    model.loglikelihood_ = ll.get() if model.use_gpu else ll

    return (model.get_ICL(), model)


def sbm_split_group(
    model: SBM, degrees: np.ndarray, index: int, indices_ones: np.array
):
    """ Given a SBM model, returns the model obtained from the split of the specified class.

    The specified class is splitted according to its median of degree.

    Parameters
    ----------
    model : sparsebm.SBM
        The model from which the merge is realized.
    degrees: numpy.ndarray
        Array that contains the degrees of the original data matrix.
    index : int
        index of the class to split.
    indices_ones : numpy.ndarray
        Indices of elements that are non-zero in the original data matrix.
    Returns
    -------
    tuple of (float, sparsebm.SBM)
        The ICL value and the model obtained from the split of the specified class.
    """
    eps = 1e-4
    model.n_clusters += 1
    t = model.tau_
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

    model.tau_ = t
    model.alpha_ = t.mean(0)
    nq = model.n_clusters

    # Transfert to GPU if necessary
    t1 = model._np.asarray(model.tau_)
    a1 = model._np.asarray(model.alpha_)
    t1_sum = t1.sum(0)

    pi = (
        t1[indices_ones[0]].reshape(-1, nq, 1)
        * t1[indices_ones[1]].reshape(-1, 1, nq)
    ).sum(0) / ((t1_sum.reshape((-1, 1)) * t1_sum) - t1.T @ t1)

    model.pi_ = pi.get() if model.use_gpu else pi
    ll = model._compute_likelihood(indices_ones, pi, a1, t1)
    model.loglikelihood_ = ll.get() if model.use_gpu else ll

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
