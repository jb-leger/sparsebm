"""
=============================================
A demo of the Latent Block Model
=============================================
This example demonstrates how to generate a dataset from the bernouilli LBM
and cluster it using the Latent Block Model.
The data is generated with the ``generate_bernouilli_LBM_dataset`` function,
and passed to the Latent Block Model. The rows and columns of the matrix
are rearranged to show the clusters found by the model.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sparsebm
from sparsebm import generate_bernouilli_LBM_dataset, LBM_bernouilli
from sparsebm.utils import reorder_rows

###
### Specifying the parameters of the dataset to generate.
###
number_of_rows = int(2 * 10 ** 3)
number_of_columns = int(number_of_rows / 2)
nb_row_clusters, nb_column_clusters = 3, 4
row_cluster_proportions = (
    np.ones(nb_row_clusters) / nb_row_clusters
)  # Here equals classe sizes
column_cluster_proportions = (
    np.ones(nb_column_clusters) / nb_column_clusters
)  # Here equals classe sizes
connection_probabilities = (
    np.array(
        [
            [0.025, 0.0125, 0.0125, 0.05],
            [0.0125, 0.025, 0.0125, 0.05],
            [0, 0.0125, 0.025, 0],
        ]
    )
) * 2

assert (
    nb_row_clusters == connection_probabilities.shape[0]
    and nb_column_clusters == connection_probabilities.shape[1]
)

###
### Generate The dataset.
###
dataset = generate_bernouilli_LBM_dataset(
    number_of_rows,
    number_of_columns,
    nb_row_clusters,
    nb_column_clusters,
    connection_probabilities,
    row_cluster_proportions,
    column_cluster_proportions,
)
graph = dataset["data"]
row_cluster_indicator = dataset["row_cluster_indicator"]
column_cluster_indicator = dataset["column_cluster_indicator"]
row_clusters_index = row_cluster_indicator.argmax(1)
column_clusters_index = column_cluster_indicator.argmax(1)

# instantiate the Latent Block Model class.
model = LBM_bernouilli(
    nb_row_clusters,  # A number of row classes must be specify. Otherwise see model selection.
    nb_column_clusters,  # A number of column classes must be specify. Otherwise see model selection.
    n_init=50,  # Specifying the number of initializations to perform.
    n_iter_early_stop=10,  # Specifying the number of EM-steps to perform on each init.
    n_init_total_run=10,  # Specifying the number inits to keep and to train until convergence.
    tol=1e-4,  # Tolerance to declare convergence
    verbosity=1,  # Either 0, 1 or 2. Higher value display more information to the user.
    gpu_number=0,  # Either None or the desired GPU index.
)
model.fit(graph)

if model.trained_successfully:
    print("Model has been trained successfully.")
    print(
        "Value of the Integrated Completed Loglikelihood is {:.4f}".format(
            model.get_ICL()
        )
    )
    row_ari = sklearn.metrics.adjusted_rand_score(
        row_clusters_index, model.row_labels
    )
    column_ari = sklearn.metrics.adjusted_rand_score(
        column_clusters_index, model.column_labels
    )
    print("Adjusted Rand index is {:.2f} for row classes".format(row_ari))
    print(
        "Adjusted Rand index is {:.2f} for column classes".format(column_ari)
    )

original_matrix = graph.copy()
reconstructed_matrix = graph.copy()
# Reordering rows
reorder_rows(original_matrix, np.argsort(row_clusters_index))
reorder_rows(reconstructed_matrix, np.argsort(model.row_labels))
# Reordering columns
original_matrix = original_matrix.transpose()
reconstructed_matrix = reconstructed_matrix.transpose()
reorder_rows(original_matrix, np.argsort(column_clusters_index))
reorder_rows(reconstructed_matrix, np.argsort(model.column_labels))
original_matrix = original_matrix.transpose()
reconstructed_matrix = reconstructed_matrix.transpose()

figure = plt.figure(1, figsize=(8, 5), constrained_layout=True)
# Plotting the original matrix.
ax1 = plt.subplot(131)
ax1.spy(graph, markersize=0.05, marker="*", c="black")
ax1.set_title("Original data matrix\n\n")
ax1.axis("off")
# Plotting the original ordered matrix.
ax2 = plt.subplot(132)
ax2.spy(original_matrix, markersize=0.05, marker="*", c="black")
ax2.set_title("Data matrix reordered \naccording to the\n original classes")
ax2.axis("off")
# Plotting the matrix reordered by the LBM.
ax3 = plt.subplot(133)
ax3.spy(reconstructed_matrix, markersize=0.05, marker="*", c="black")
ax3.set_title(
    "Data matrix reordered \naccording to the\n classes given by the LBM"
)
ax3.axis("off")
plt.show()
