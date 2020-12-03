import numpy as np
import scipy as sp
import scipy.sparse
import progressbar
from typing import Optional

_number_of_rows_default = 10 ** 3
_number_of_columns_default = int(_number_of_rows_default / 2)
_nb_row_clusters_default = np.random.randint(3, 6)
_nb_column_clusters_default = np.random.randint(3, 6)
_connection_probabilities_default_lbm = np.random.rand(
    _nb_row_clusters_default, _nb_column_clusters_default
)
c = 0.03 / _connection_probabilities_default_lbm.mean()
_connection_probabilities_default_lbm *= c
_row_cluster_proportions_default = (
    np.ones(_nb_row_clusters_default) / _nb_row_clusters_default
)
_column_cluster_proportions_default = (
    np.ones(_nb_column_clusters_default) / _nb_column_clusters_default
)


def generate_LBM_dataset(
    number_of_rows: Optional[int] = _number_of_rows_default,
    number_of_columns: Optional[int] = _number_of_columns_default,
    nb_row_clusters: Optional[int] = _nb_row_clusters_default,
    nb_column_clusters: Optional[int] = _nb_column_clusters_default,
    connection_probabilities: Optional[
        np.ndarray
    ] = _connection_probabilities_default_lbm,
    row_cluster_proportions: Optional[
        np.ndarray
    ] = _row_cluster_proportions_default,
    column_cluster_proportions: Optional[
        np.ndarray
    ] = _column_cluster_proportions_default,
    verbosity: Optional[int] = 1,
) -> dict:
    """ Generate a sparse bipartite graph with Latent Block Models.

    Parameters
    ----------
    number_of_rows : int, optional, default : 1000
        The number of nodes of type (1).
    number_of_columns : int, optional, default : 500
        The number of nodes of type (2).
    nb_row_clusters : int, optional, default : random between 3 and 5
        The number of classes of nodes of type (1).
    nb_column_clusters : int, default : random between 3 and 5
        The number of classes of nodes of type (2).
    connection_probabilities : np.ndarray, optional, default : random such as sparsity is 0.03
        The probability of having an edge between the classes.
    row_cluster_proportions : np.ndarray, optional, default : balanced
        Proportion of the classes of nodes of type (1).
    column_cluster_proportions : np.ndarray, optional, default : balanced
        Proportion of the classes of nodes of type (2).
    verbosity : int, optional, default : 1
        Display information during the generation process.
    Returns
    -------
    dataset: dict
        The generated dataset. Keys contain 'data', the scipy.sparse.coo
        adjacency matrix; 'row_cluster_indicator' and 'column_cluster_indicator'
        the np.ndarray of class membership of nodes.
    """
    try:
        if verbosity > 0:
            print("---------- START Graph Generation ---------- ")
            bar = progressbar.ProgressBar(
                max_value=nb_row_clusters * nb_column_clusters,
                widgets=[
                    progressbar.SimpleProgress(),
                    " Generating block: ",
                    " [",
                    progressbar.Percentage(),
                    " ] ",
                    progressbar.Bar(),
                    " [ ",
                    progressbar.Timer(),
                    " ] ",
                ],
                redirect_stdout=True,
            ).start()
        row_cluster_indicator = np.random.multinomial(
            1, row_cluster_proportions.flatten(), size=number_of_rows
        )
        column_cluster_indicator = np.random.multinomial(
            1, column_cluster_proportions.flatten(), size=number_of_columns
        )
        row_classes = [
            row_cluster_indicator[:, q].nonzero()[0]
            for q in range(nb_row_clusters)
        ]
        col_classes = [
            column_cluster_indicator[:, l].nonzero()[0]
            for l in range(nb_column_clusters)
        ]

        rows = np.array([])
        cols = np.array([])
        for i, (q, l) in enumerate(
            [
                (i, j)
                for i in range(nb_row_clusters)
                for j in range(nb_column_clusters)
            ]
        ):
            if verbosity > 0:
                bar.update(i)
            n1, n2 = row_classes[q].size, col_classes[l].size
            nnz = np.random.binomial(n1 * n2, connection_probabilities[q, l])
            if nnz > 0:
                row = np.random.choice(row_classes[q], size=2 * nnz)
                col = np.random.choice(col_classes[l], size=2 * nnz)
                row_col_unique = np.unique(np.stack((row, col), 1), axis=0)
                while row_col_unique.shape[0] < nnz:
                    row = np.random.choice(row_classes[q], size=2 * nnz)
                    col = np.random.choice(col_classes[l], size=2 * nnz)
                    row_col_unique = np.unique(np.stack((row, col), 1), axis=0)
                np.random.shuffle(row_col_unique)
                rows = np.concatenate((rows, row_col_unique[:nnz, 0]))
                cols = np.concatenate((cols, row_col_unique[:nnz, 1]))

        graph = scipy.sparse.coo_matrix(
            (np.ones(rows.size), (rows, cols)),
            shape=(number_of_rows, number_of_columns),
        )
        if verbosity > 0:
            bar.finish()

    except KeyboardInterrupt:
        return None
    finally:
        if verbosity > 0:
            bar.finish()

    dataset = {
        "data": graph,
        "row_cluster_indicator": row_cluster_indicator,
        "column_cluster_indicator": column_cluster_indicator,
    }

    return dataset


_number_of_nodes_default = 10 ** 3
_nb_clust_default = np.random.randint(3, 6)
_connection_probabilities_default = np.random.rand(
    _nb_clust_default, _nb_clust_default
)
c = 0.03 / _connection_probabilities_default.mean()
_connection_probabilities_default *= c
_cluster_proportions_default = np.ones(_nb_clust_default) / _nb_clust_default


def generate_SBM_dataset(
    number_of_nodes: Optional[int] = _number_of_nodes_default,
    number_of_clusters: Optional[int] = _nb_clust_default,
    connection_probabilities: Optional[
        np.ndarray
    ] = _connection_probabilities_default,
    cluster_proportions: Optional[np.ndarray] = _cluster_proportions_default,
    symmetric: Optional[bool] = False,
    verbosity: Optional[int] = 1,
) -> dict:
    """ Generate a sparse graph with Stochastic Block Models.

    Parameters
    ----------
    number_of_nodes : int, optional, default : 1000
        The number of nodes.
    number_of_clusters : int, optional, default : random between 3 and 5
        The number of classes of nodes.
    connection_probabilities : np.ndarray, optional, default : random such as sparsity is 0.03
        The probability of having an edge between the classes.
    cluster_proportions : np.ndarray, optional, default : balanced
        Proportion of the classes of nodes.
    symmetric : bool, optional, default : False
        Specify if the generated adjacency matrix is symmetric.
    verbosity : int, optional, default : 1
        Display information during the generation process.
    Returns
    -------
    dataset: dict
        The generated dataset. Keys contain 'data', the scipy.sparse.coo
        adjacency matrix; 'cluster_indicator' the np.ndarray of class
        membership of nodes.
    """
    try:
        if verbosity > 0:
            print("---------- START Graph Generation ---------- ")
            bar = progressbar.ProgressBar(
                max_value=number_of_clusters ** 2,
                widgets=[
                    progressbar.SimpleProgress(),
                    " Generating block: ",
                    " [",
                    progressbar.Percentage(),
                    " ] ",
                    progressbar.Bar(),
                    " [ ",
                    progressbar.Timer(),
                    " ] ",
                ],
                redirect_stdout=True,
            ).start()
        cluster_indicator = np.random.multinomial(
            1, cluster_proportions.flatten(), size=number_of_nodes
        )
        classes = [
            cluster_indicator[:, q].nonzero()[0]
            for q in range(number_of_clusters)
        ]

        rows = np.array([])
        cols = np.array([])
        for i, (q, l) in enumerate(
            [
                (i, j)
                for i in range(number_of_clusters)
                for j in range(number_of_clusters)
            ]
        ):
            if verbosity > 0:
                bar.update(i)
            n1, n2 = classes[q].size, classes[l].size

            if connection_probabilities[q, l] >= 0.25:
                for id in classes[q]:
                    nb_ones = np.random.binomial(
                        classes[l].size, connection_probabilities[q, l]
                    )
                    col = np.random.choice(classes[l], nb_ones, replace=False)
                    row = np.ones_like(col) * id
                    row_col_unique = np.unique(np.stack((row, col), 1), axis=0)
                    np.random.shuffle(row_col_unique)
                    rows = np.concatenate((rows, row_col_unique[:, 0]))
                    cols = np.concatenate((cols, row_col_unique[:, 1]))
            else:
                nnz = np.random.binomial(
                    n1 * n2, connection_probabilities[q, l]
                )
                if nnz > 0:
                    row = np.random.choice(classes[q], size=2 * nnz)
                    col = np.random.choice(classes[l], size=2 * nnz)
                    row_col_unique = np.unique(np.stack((row, col), 1), axis=0)
                    while row_col_unique.shape[0] < nnz:
                        row = np.random.choice(classes[q], size=2 * nnz)
                        col = np.random.choice(classes[l], size=2 * nnz)
                        row_col_unique = np.unique(
                            np.stack((row, col), 1), axis=0
                        )
                    np.random.shuffle(row_col_unique)
                    rows = np.concatenate((rows, row_col_unique[:nnz, 0]))
                    cols = np.concatenate((cols, row_col_unique[:nnz, 1]))

        inserted = np.stack((rows, cols), axis=1)
        if symmetric:
            inserted = inserted[inserted[:, 0] < inserted[:, 1]]
            inserted = np.concatenate((inserted, inserted[:, [1, 0]]))
        else:
            inserted = inserted[inserted[:, 0] != inserted[:, 1]]

        graph = scipy.sparse.coo_matrix(
            (np.ones(inserted[:, 0].size), (inserted[:, 0], inserted[:, 1])),
            shape=(number_of_nodes, number_of_nodes),
        )
        if verbosity > 0:
            bar.finish()

    except KeyboardInterrupt:
        return None
    finally:
        if verbosity > 0:
            bar.finish()

    return {"data": graph, "cluster_indicator": cluster_indicator}
