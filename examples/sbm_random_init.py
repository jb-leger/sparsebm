"""
=============================================
A demo of the Stochastic Block Model
=============================================
This example demonstrates how to generate a dataset from the bernouilli SBM
and cluster it using the Stochastic Block Model.
The data is generated with the ``generate_bernouilli_SBM_dataset`` function,
and passed to the Stochastic Block Model. The rows and columns of the matrix
are rearranged to show the clusters found by the model.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
import sparsebm
from sparsebm import generate_bernouilli_SBM_dataset, SBM_bernouilli
from sparsebm.utils import reorder_rows, ARI

# Specifying the parameters of the dataset to generate.
number_of_nodes = 10 ** 3
number_of_clusters = 4
cluster_proportions = (
    np.ones(number_of_clusters) / number_of_clusters
)  # Here equals classe sizes
connection_probabilities = (
    np.array(
        [
            [0.05, 0.018, 0.006, 0.0307],
            [0.018, 0.037, 0, 0],
            [0.006, 0, 0.055, 0.012],
            [0.0307, 0, 0.012, 0.043],
        ]
    )
    * 2
)  # The probability of link between the classes. Here symetric.
assert (
    number_of_clusters
    == connection_probabilities.shape[0]
    == connection_probabilities.shape[1]
)

# Generate The dataset.
dataset = generate_bernouilli_SBM_dataset(
    number_of_nodes,
    number_of_clusters,
    connection_probabilities,
    cluster_proportions,
    symetric=False,
)
graph = dataset["data"]
cluster_indicator = dataset["cluster_indicator"]
clusters_index = cluster_indicator.argmax(1)

# instantiate the Stochastic Block Model class.
model = SBM_bernouilli(
    number_of_clusters,  # A number of classes must be specify. Otherwise see model selection.
    n_init=50,  # Specifying the number of initializations to perform.
    n_iter_early_stop=30,  # Specifying the number of EM-steps to perform on each init.
    n_init_total_run=10,  # Specifying the number inits to keep and to train until convergence.
    verbosity=1,  # Either 0, 1 or 2. Higher value display more information to the user.
)
model.fit(graph, symetric=True)

if model.trained_successfully:
    print("Model has been trained successfully.")
    print(
        "Value of the Integrated Completed Loglikelihood is {:.4f}".format(
            model.get_ICL()
        )
    )
    ari = ARI(clusters_index, model.labels)
    print("Adjusted Rand index is {:.2f}".format(ari))

#
original_matrix = graph.copy()
reconstructed_matrix = graph.copy()
reorder_rows(original_matrix, np.argsort(clusters_index))
reorder_rows(reconstructed_matrix, np.argsort(model.labels))
original_matrix = original_matrix.transpose()
reconstructed_matrix = reconstructed_matrix.transpose()
reorder_rows(original_matrix, np.argsort(clusters_index))
reorder_rows(reconstructed_matrix, np.argsort(model.labels))
original_matrix = original_matrix.transpose()
reconstructed_matrix = reconstructed_matrix.transpose()

figure, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(8, 5), constrained_layout=True
)
# Plotting the original matrix.
ax1.spy(graph, markersize=0.05, marker="*", c="black")
ax1.set_title("Original data matrix")
ax1.axis("off")
# Plotting the original ordered matrix.
ax2.spy(original_matrix, markersize=0.05, marker="*", c="black")
ax2.set_title("Data matrix reordered \naccording to the\noriginal classes")
ax2.axis("off")
# Plotting the matrix reordered by the SBM.
ax3.spy(reconstructed_matrix, markersize=0.05, marker="*", c="black")
ax3.set_title(
    "Data matrix reordered \naccording to the\nclasses given by the SBM"
)
ax3.axis("off")
plt.show()
