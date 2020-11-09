import numpy as np
import scipy as sp
import scipy.sparse


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


def generate_bernouilli_LBM_dataset(
    number_of_rows,
    number_of_columns,
    nb_row_clusters,
    nb_column_clusters,
    connection_probabilities,
    row_cluster_proportions,
    column_cluster_proportions,
):
    print("Start generating graph, it might take a while...")
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

    inserted = set()
    for q in range(nb_row_clusters):
        print(q)
        for l in range(nb_column_clusters):
            if connection_probabilities[q, l] >= 0.25:
                # rejection algo not effecient
                for i in row_classes[q]:
                    nb_ones = np.random.binomial(
                        col_classes[l].size, connection_probabilities[q, l]
                    )
                    trucs = np.random.choice(
                        col_classes[l], nb_ones, replace=False
                    )
                    inserted.update((i, j) for j in trucs)
            else:
                nb_ones = np.random.binomial(
                    row_classes[q].size * col_classes[l].size,
                    connection_probabilities[q, l],
                )
                c = 0
                while c < nb_ones:
                    i = np.random.choice(row_classes[q])
                    j = np.random.choice(col_classes[l])
                    if (i, j) not in inserted:
                        inserted.add((i, j))
                        c += 1

    X = sp.sparse.coo_matrix(
        (
            np.ones(len(inserted)),
            ([i for i, j in inserted], [j for i, j in inserted]),
        ),
        (number_of_rows, number_of_columns),
    )

    return {
        "data": X,
        "row_cluster_indicator": row_cluster_indicator,
        "column_cluster_indicator": column_cluster_indicator,
    }
