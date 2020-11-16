import numpy as np
import scipy as sp
import scipy.sparse
import progressbar
from typing import Optional


def generate_bernouilli_LBM_dataset(
    number_of_rows: int,
    number_of_columns: int,
    nb_row_clusters: int,
    nb_column_clusters: int,
    connection_probabilities: np.ndarray,
    row_cluster_proportions: np.ndarray,
    column_cluster_proportions: np.ndarray,
    verbosity: Optional[int] = 1,
) -> dict:
    """ Generate a sparse bipartite graph with Latent Block Models.

    Parameters
    ----------
    number_of_rows : int
        The number of nodes of type (1).
    number_of_columns : int
        The number of nodes of type (2).
    nb_row_clusters : int
        The number of classes of nodes of type (1).
    nb_column_clusters : int
        The number of classes of nodes of type (2).
    connection_probabilities : np.ndarray
        The probability of having an edge between the classes.
    row_cluster_proportions : np.ndarray
        Proportion of the classes of nodes of type (1).
    column_cluster_proportions : np.ndarray
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


def generate_bernouilli_SBM_dataset(
    number_of_nodes,
    number_of_clusters,
    connection_probabilities,
    cluster_proportions,
    symetric=False,
):
    print("Start generating graph, it might take a while...")
    cluster_indicator = np.random.multinomial(
        1, cluster_proportions.flatten(), size=number_of_nodes
    )
    classes = [
        cluster_indicator[:, q].nonzero()[0] for q in range(number_of_clusters)
    ]

    inserted = set()
    for q in range(number_of_clusters):
        for l in range(number_of_clusters):
            if connection_probabilities[q, l] >= 0.25:
                # rejection algo not effecient
                for i in classes[q]:
                    nb_ones = np.random.binomial(
                        classes[l].size, connection_probabilities[q, l]
                    )
                    trucs = np.random.choice(
                        classes[l], nb_ones, replace=False
                    )
                    inserted.update((i, j) for j in trucs)
            else:
                nb_ones = np.random.binomial(
                    classes[q].size * classes[l].size,
                    connection_probabilities[q, l],
                )
                c = 0
                while c < nb_ones:
                    i = np.random.choice(classes[q])
                    j = np.random.choice(classes[l])
                    if (i, j) not in inserted:
                        inserted.add((i, j))
                        c += 1
    if symetric:
        inserted = [(i, j) for (i, j) in inserted if i < j]
        inserted.extend([(j, i) for (i, j) in inserted])
    else:
        inserted = [(i, j) for (i, j) in inserted if i != j]
    X = sp.sparse.coo_matrix(
        (
            np.ones(len(inserted)),
            ([i for i, j in inserted], [j for i, j in inserted]),
        ),
        (number_of_nodes, number_of_nodes),
    )

    return {"data": X, "cluster_indicator": cluster_indicator}
