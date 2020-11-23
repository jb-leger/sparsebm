import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sparsebm
from sparsebm.utils import reorder_rows, ARI, CARI
import glob
import pickle
import time
import cupy
import experiments.lbm_not_sparse

# f_prefix = "500_250"
# f_prefix = "1000_500"
# f_prefix = "1500_750"
# f_prefix = "2000_1000"
# f_prefix = "2500_1250"
# f_prefix = "3000_1500"
# f_prefix = "5000_2500"
# f_prefix = "10000_5000"
# f_prefix = "15000_7500"
# f_prefix = "20000_10000"
# f_prefix = "40000_20000"
f_prefix = "80000_40000"
# gpu_index = 0
# gpu_index = 1
# gpu_index = 2
# gpu_index = 3
# gpu_index = 4
# gpu_index = 5
# gpu_index = 6
gpu_index = 7

dataset_files = glob.glob(
    "./experiments/data/sparsity_fixed/" + f_prefix + "_*.pkl"
)
not_sparse = False

nb_row_clusters, nb_column_clusters = 3, 4
n_init = 100  # Specifying the number of initializations to perform.
n_iter_early_stop = (
    20
)  # Specifying the number of EM-steps to perform on each init.
n_init_total_run = (
    10
)  # Specifying the number inits to keep and to train until convergence.


def train_with_both_model(dataset_file, gpu_index):
    print(dataset_file)

    results_files_already_done = glob.glob(
        "./experiments/results/sparsity_fixed/*.pkl"
    )
    if (
        "./experiments/results/sparsity_fixed/" + dataset_file.split("/")[-1]
        in results_files_already_done
    ):
        print("Already Done")
        return None

    dataset = pickle.load(open(dataset_file, "rb"))
    graph = dataset["data"]
    row_cluster_indicator = dataset["row_cluster_indicator"]
    column_cluster_indicator = dataset["column_cluster_indicator"]
    row_clusters_index = row_cluster_indicator.argmax(1)
    column_clusters_index = column_cluster_indicator.argmax(1)
    exponent = dataset["exponent"]
    seed = np.random.randint(0, 10000)

    if not_sparse:
        # instantiate the Latent Block Model class.
        print("Training not sparse")
        model = experiments.lbm_not_sparse.LBM_bernouilli_not_sparse(
            nb_row_clusters,  # A number of row classes must be specify. Otherwise see model selection.
            nb_column_clusters,  # A number of column classes must be specify. Otherwise see model selection.
            n_init=n_init,  # Specifying the number of initializations to perform.
            n_iter_early_stop=n_iter_early_stop,  # Specifying the number of EM-steps to perform on each init.
            n_init_total_run=n_init_total_run,  # Specifying the number inits to keep and to train until convergence.
            verbosity=1,  # Either 0, 1 or 2. Higher value display more information to the user.
            gpu_index=gpu_index,
        )
        cupy.random.seed(seed)
        np.random.seed(seed)

        start_time_not_sparse = time.time()
        model.fit(np.array(graph.todense()))
        end_time_not_sparse = time.time() - start_time_not_sparse

        row_ari_not_sparse = ARI(row_clusters_index, model.row_labels)
        column_ari_not_sparse = ARI(column_clusters_index, model.column_labels)
        co_ari_not_sparse = CARI(
            row_clusters_index,
            column_clusters_index,
            model.row_labels,
            model.column_labels,
        )
        print(end_time_not_sparse)
        print(f"coari not sparse {co_ari_not_sparse}")

    else:
        end_time_not_sparse = None
        row_ari_not_sparse = None
        column_ari_not_sparse = None
        co_ari_not_sparse = None

    print("Training sparse")
    # instantiate the Latent Block Model class.
    model2 = sparsebm.lbm.LBM_bernouilli(
        nb_row_clusters,  # A number of row classes must be specify. Otherwise see model selection.
        nb_column_clusters,  # A number of column classes must be specify. Otherwise see model selection.
        n_init=n_init,  # Specifying the number of initializations to perform.
        n_iter_early_stop=n_iter_early_stop,  # Specifying the number of EM-steps to perform on each init.
        n_init_total_run=n_init_total_run,  # Specifying the number inits to keep and to train until convergence.
        verbosity=1,  # Either 0, 1 or 2. Higher value display more information to the user.
        gpu_index=gpu_index,
    )
    cupy.random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    model2.fit(graph)
    end_time = time.time() - start_time

    row_ari = ARI(row_clusters_index, model2.row_labels)
    column_ari = ARI(column_clusters_index, model2.column_labels)
    co_ari = CARI(
        row_clusters_index,
        column_clusters_index,
        model2.row_labels,
        model2.column_labels,
    )
    print(end_time)
    print(f"coari {co_ari}")

    results = {
        "dataset_file": dataset_file,
        "seed": seed,
        "exponent": exponent,
        #
        "row_ari_not_sparse": row_ari_not_sparse,
        "row_ari": row_ari,
        #
        "column_ari_not_sparse": column_ari_not_sparse,
        "column_ari": column_ari,
        #
        "co_ari_not_sparse": co_ari_not_sparse,
        "co_ari": co_ari,
        #
        "end_time_not_sparse": end_time_not_sparse,
        "end_time": end_time,
        #
        "model": {
            "pi": model2.pi_,
            "alpha_1": model2.alpha_1_,
            "alpha_2": model2.alpha_2_,
            "tau_1": model2.tau_1_,
            "tau_2": model2.tau_2_,
        },
    }

    if not_sparse:
        results["model_not_sparse"] = (
            {
                "pi": model.pi_,
                "alpha_1": model.alpha_1_,
                "alpha_2": model.alpha_2_,
                "tau_1": model.tau_1_,
                "tau_2": model.tau_2_,
            },
        )
    pickle.dump(
        results,
        open(
            "./experiments/results/sparsity_fixed/"
            + dataset_file.split("/")[-1],
            "wb",
        ),
    )


for dataset_file in dataset_files:
    train_with_both_model(dataset_file, gpu_index)

exit(1)
