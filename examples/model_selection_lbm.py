"""
=================================================================
A demo of the Model Selection applied to the Latent Block Model
=================================================================
This example demonstrates how to generate a dataset from the bernouilli LBM
and cluster it using the Latent Block Model without a prior knowledge on the
number of classes. The data is generated with the
``generate_bernouilli_LBM_dataset`` function, and passed to the ModelSelection
class. The rows and columns of the matrix are rearranged to show the clusters
found by the model.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sparsebm
from sparsebm import generate_bernouilli_LBM_dataset, ModelSelection
from sparsebm.utils import reorder_rows

###
### Specifying the parameters of the dataset to generate.
###
number_of_rows = int(10 ** 3)
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

# instantiate the ModelSelection class.
lbm_model_selection = ModelSelection(
    graph,
    model_type="LBM",  # Either 'LBM' or 'SBM', the model used to cluster the data.
    plot=True,  # display illustration of the model selection algorithm.
)

lbm_selected = lbm_model_selection.fit()

if lbm_selected.trained_successfully:
    print("Model has been trained successfully.")
    print(
        "Value of the Integrated Completed Loglikelihood is {:.4f}".format(
            lbm_selected.get_ICL()
        )
    )
    print("The original number of row classes was {}".format(nb_row_clusters))
    print(
        "The model selection picked {} row classes".format(
            lbm_selected.n_row_clusters
        )
    )
    print(
        "The original number of column classes was {}".format(
            nb_column_clusters
        )
    )
    print(
        "The model selection picked {} column classes".format(
            lbm_selected.n_column_clusters
        )
    )
    row_ari = sklearn.metrics.adjusted_rand_score(
        row_clusters_index, lbm_selected.row_labels
    )
    column_ari = sklearn.metrics.adjusted_rand_score(
        column_clusters_index, lbm_selected.column_labels
    )
    print("Adjusted Rand index is {:.2f} for row classes".format(row_ari))
    print(
        "Adjusted Rand index is {:.2f} for column classes".format(column_ari)
    )

original_matrix = graph.copy()
reconstructed_matrix = graph.copy()
# Reordering rows
reorder_rows(original_matrix, np.argsort(row_clusters_index))
reorder_rows(reconstructed_matrix, np.argsort(lbm_selected.row_labels))
# Reordering columns
original_matrix = original_matrix.transpose()
reconstructed_matrix = reconstructed_matrix.transpose()
reorder_rows(original_matrix, np.argsort(column_clusters_index))
reorder_rows(reconstructed_matrix, np.argsort(lbm_selected.column_labels))
original_matrix = original_matrix.transpose()
reconstructed_matrix = reconstructed_matrix.transpose()

plt.clf()
figure = plt.figure(1, figsize=(8, 5), constrained_layout=True)
# Plotting the original matrix.
ax1 = plt.subplot(131)
ax1.spy(graph, markersize=0.05, marker="*", c="black")
ax1.set_title("Original data matrix")
ax1.axis("off")
# Plotting the original ordered matrix.
ax2 = plt.subplot(132)
ax2.spy(original_matrix, markersize=0.05, marker="*", c="black")
ax2.set_title("Data matrix reordered \naccording to the \noriginal classes")
ax2.axis("off")
# Plotting the matrix reordered by the LBM.
ax3 = plt.subplot(133)
ax3.spy(reconstructed_matrix, markersize=0.05, marker="*", c="black")
ax3.set_title(
    "Data matrix reordered \naccording to the \nclasses given by the LBM"
)
ax3.axis("off")
plt.show()
