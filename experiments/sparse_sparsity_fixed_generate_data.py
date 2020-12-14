import numpy as np
import sparsebm
from sparsebm import generate_LBM_dataset, ModelSelection
from sparsebm.utils import reorder_rows, ARI, CARI
import scipy.sparse as ss
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--repeat",
    type=int,
    help="Number of arrays for each dimension to be generated.",
    required=False,
    default=100,
)
args = vars(parser.parse_args())
nbtt = args["repeat"]

if not os.path.exists("./experiments/data"):
    os.makedirs("./experiments/data")
if not os.path.exists("./experiments/data/sparsity_fixed"):
    os.makedirs("./experiments/data/sparsity_fixed")

###
### Specifying the parameters of the dataset to generate.
###
nb_row_clusters, nb_column_clusters = 3, 4
row_cluster_proportions = (
    np.ones(nb_row_clusters) / nb_row_clusters
)  # Here equals classe sizes
column_cluster_proportions = (
    np.ones(nb_column_clusters) / nb_column_clusters
)  # Here equals classe sizes

e = 0.25
exponent = 5
connection_probabilities = (
    np.array([[4 * e, e, e, e * 2], [e, e, e, e], [2 * e, e, 2 * e, 2 * e]])
    / 2 ** exponent
)


###
### Generate The dataset.
###
import pickle

number_of_rows = np.array(
    [
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        5000,
        10000,
        15000,
        20000,
        40000,
        80000,
    ]
)
number_of_columns = (number_of_rows / 2).astype(int)
for n1, n2 in np.stack((number_of_rows, number_of_columns), 1):
    print("Sizes {}-{}".format(n1, n2))
    for i in range(nbtt):
        print("Generate dataset {}/{}".format(i, nbtt))
        dataset = generate_LBM_dataset(
            n1,
            n2,
            nb_row_clusters,
            nb_column_clusters,
            connection_probabilities,
            row_cluster_proportions,
            column_cluster_proportions,
            verbosity=0,
        )
        dataset["connection_probabilities"] = connection_probabilities
        dataset["n1"] = n1
        dataset["n2"] = n2
        dataset["exponent"] = exponent
        fname = str(n1) + "_" + str(n2) + "_" + str(i) + ".pkl"
        pickle.dump(
            dataset, open("./experiments/data/sparsity_fixed/" + fname, "wb")
        )
