import os

NB_THREAD_MAX = 1
os.environ["MKL_NUM_THREADS"] = str(NB_THREAD_MAX)
os.environ["NUMEXPR_NUM_THREADS"] = str(NB_THREAD_MAX)
os.environ["OMP_NUM_THREADS"] = str(NB_THREAD_MAX)

from sparsebm import LBM_bernouilli
from sparsebm.utils import CARI
import numpy as np
from sparsebm import LBM_bernouilli
from resource import getrusage as resource_usage, RUSAGE_SELF
from time import time as timestamp
import glob
import pickle
import argparse

try:
    import cupy

    _CUPY_INSTALLED = True
except ImportError:
    _CUPY_INSTALLED = False


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--programs",
    nargs="+",
    help="List of lib to use. Example 'sparsebm', 'blockmodels', 'blockcluster'",
    required=False,
    default=["sparsebm", "blockmodels", "blockcluster"],
)
parser.add_argument(
    "-s",
    "--size",
    type=int,
    help="Limit size of graph (first dimension). Exemple: 1000",
    required=True,
    default=np.inf,
)

parser.add_argument(
    "-g",
    "--use_gpu",
    help="Specify if a GPU should be used.",
    default=False,
    required=False,
    type=bool,
)

args = vars(parser.parse_args())

use_sp = True if "sparsebm" in args["programs"] else False
use_bm = True if "blockmodels" in args["programs"] else False
use_bc = True if "blockcluster" in args["programs"] else False


if args["use_gpu"]:
    use_gpu = True
else:
    use_gpu = False

if use_bm or use_bc:
    import rpy2.robjects as ro
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr

    rpy2.robjects.numpy2ri.activate()
if use_bm:
    blockmodels = importr("blockmodels")
if use_bc:
    blockcluster = importr("blockcluster")


f_prefix_list = [
    "500_250",
    "1000_500",
    "1500_750",
    "2000_1000",
    "2500_1250",
    "3000_1500",
    "5000_2500",
    "10000_5000",
    "15000_7500",
    "20000_10000",
    "40000_20000",
    "80000_40000",
]

dataset_files = [
    glob.glob("./experiments/data/sparsity_fixed/" + f_prefix + "_*.pkl")
    for f_prefix in f_prefix_list
    if int(f_prefix.split("_")[0]) <= args["size"]
]

dataset_files = [b for a in dataset_files for b in a]
dataset_files = np.array(dataset_files)
np.random.shuffle(dataset_files)


results_folder = "./experiments/results/benchmark_libraries/size_growing/"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


def train_with_sparsebm(
    dataset_file,
    graph,
    nb_row_clusters,
    nb_column_clusters,
    row_clusters_index,
    column_clusters_index,
    use_gpu=False,
):
    results_files_already_done = glob.glob(results_folder + "*.pkl")
    save_f = (
        results_folder + dataset_file.split("/")[-1].split(".")[0] + "_sp.pkl"
    )
    if use_gpu:
        save_f = (
            results_folder
            + dataset_file.split("/")[-1].split(".")[0]
            + "_sp_gpu.pkl"
        )
    if save_f in results_files_already_done:
        print("Already Done")
        return None
    model = LBM_bernouilli(
        nb_row_clusters,
        nb_column_clusters,
        n_init=100,
        n_iter_early_stop=10,
        n_init_total_run=1,
        max_iter=5000,
        verbosity=1,
        use_gpu=use_gpu,
    )
    start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)
    model.fit(graph)
    end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()
    co_ari = CARI(
        row_clusters_index,
        column_clusters_index,
        model.row_labels,
        model.column_labels,
    )
    icl = model.get_ICL()
    results = {
        "lib": "sparsebm",
        "gpu": use_gpu,
        "n1": graph.shape[0],
        "n2": graph.shape[1],
        "nq": nb_row_clusters,
        "nl": nb_column_clusters,
        "dataset_file": dataset_file,
        "icl": icl,
        "cari": co_ari,
        "real": end_time - start_time,
        "sys": end_resources.ru_stime - start_resources.ru_stime,
        "user": end_resources.ru_utime - start_resources.ru_utime,
    }
    print(f'SparseBM tt time {results["user"]+results["sys"]}')
    pickle.dump(results, open(save_f, "wb"))
    return results


def train_with_blockmodels(
    dataset_file,
    graph,
    nb_row_clusters,
    nb_column_clusters,
    row_clusters_index,
    column_clusters_index,
):
    results_files_already_done = glob.glob(results_folder + "*.pkl")
    if (
        results_folder + dataset_file.split("/")[-1].split(".")[0] + "_bm.pkl"
        in results_files_already_done
    ):
        print("Already Done")
        return None

    print("blockmodels :")
    # Convert sparse matrix to R matrix.
    n1, n2 = graph.shape
    B = graph.todense()
    nr, nc = B.shape
    Br = ro.r.matrix(B, nrow=nr, ncol=nc)
    network = robjects.ListVector({"adjacency": Br})

    model = LBM_bernouilli(
        nb_row_clusters,
        nb_column_clusters,
        n_init=1,
        n_iter_early_stop=1,
        n_init_total_run=1,
        max_iter=1,
        verbosity=0,
    )
    model.fit(graph)
    init_list = []
    for _ in range(100):
        _, _, tau_1_init, tau_2_init, _ = model._init_bernouilli_LBM_random(
            n1, n2, nb_row_clusters, nb_column_clusters, graph.nnz
        )
        nr, nc = tau_1_init.shape
        t1_init = ro.r.matrix(tau_1_init, nrow=nr, ncol=nc)
        nr, nc = tau_2_init.shape
        t2_init = ro.r.matrix(tau_2_init, nrow=nr, ncol=nc)
        init_list.append(robjects.ListVector({"Z1": t1_init, "Z2": t2_init}))

    start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)
    best_icl = -np.inf
    best_init = None
    for i, init in enumerate(init_list):
        print(f"Init {i}/{len(init_list)}", end="\r")
        results = blockmodels.dispatcher(
            "LBM", init, "bernoulli", network, False
        )
        icl_or_ll = results[2][0]
        if icl_or_ll > best_icl:
            best_init = icl_or_ll
            best_init = init
    print("\n Start training best")
    results = blockmodels.dispatcher(
        "LBM", best_init, "bernoulli", network, True
    )
    print("End training best")
    end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()
    icl = results[2][0]
    res_tau_1 = np.array(results[0][0])
    res_tau_2 = np.array(results[0][2])
    co_ari = CARI(
        row_clusters_index,
        column_clusters_index,
        res_tau_1.argmax(1),
        res_tau_2.argmax(1),
    )
    results = {
        "lib": "blockmodels",
        "n1": graph.shape[0],
        "n2": graph.shape[1],
        "nq": nb_row_clusters,
        "nl": nb_column_clusters,
        "dataset_file": dataset_file,
        "icl": icl,
        "cari": co_ari,
        "real": end_time - start_time,
        "sys": end_resources.ru_stime - start_resources.ru_stime,
        "user": end_resources.ru_utime - start_resources.ru_utime,
    }
    print(f'Blockmodels tt time {results["user"]+results["sys"]}')
    pickle.dump(
        results,
        open(
            results_folder
            + dataset_file.split("/")[-1].split(".")[0]
            + "_bm.pkl",
            "wb",
        ),
    )
    return results


def train_with_blockcluster(
    dataset_file,
    graph,
    nb_row_clusters,
    nb_column_clusters,
    row_clusters_index,
    column_clusters_index,
):
    results_files_already_done = glob.glob(results_folder + "*.pkl")
    if (
        results_folder + dataset_file.split("/")[-1].split(".")[0] + "_bc.pkl"
        in results_files_already_done
    ):
        print("Already Done")
        return None

    print("BlockCluster :")
    # Convert sparse matrix to R matrix.
    B = graph.todense()
    nr, nc = B.shape
    Br = ro.r.matrix(B, nrow=nr, ncol=nc)
    network = robjects.ListVector({"adjacency": Br})
    # initmethod Method to initialize model parameters. The valid values are "cemInitStep", "emInitStep" and "randomInit"
    #  nbiterationsxem : Number of EM iterations used during xem step. Default value is 50.
    # nbinitmax : Maximal number initialization to try. Default value is 100
    # nbinititerations : Number of Global iterations used in initialization step. Default value is 10.
    # initepsilon : Tolerance value used while initialization. Default value is 1e-2.
    # nbxem : Number of xem steps. Default value is 5.
    strategy = blockcluster.coclusterStrategy(
        initmethod="randomInit",
        nbinitmax=100,
        nbinititerations=10,
        # nbiterationsXEM=500,
        # initepsilon=1e-2,
    )
    start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)
    results = blockcluster.cocluster(
        Br,
        "binary",
        nbcocluster=robjects.IntVector([nb_row_clusters, nb_column_clusters]),
        nbCore=4,
        strategy=strategy,
    )
    end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()
    rowclass = np.array(results.slots["rowclass"])
    colclass = np.array(results.slots["colclass"])
    icl = results.slots["ICLvalue"][0]
    co_ari = CARI(
        row_clusters_index, column_clusters_index, rowclass, colclass
    )
    """Return `real`, `sys` and `user` elapsed time, like UNIX's command `time`
    You can calculate the amount of used CPU-time used by summing `user`
    and `sys`. `real` is just like the wall clock.
    """
    results = {
        "lib": "blockcluster",
        "n1": graph.shape[0],
        "n2": graph.shape[1],
        "nq": nb_row_clusters,
        "nl": nb_column_clusters,
        "dataset_file": dataset_file,
        "icl": icl,
        "cari": co_ari,
        "real": end_time - start_time,
        "sys": end_resources.ru_stime - start_resources.ru_stime,
        "user": end_resources.ru_utime - start_resources.ru_utime,
    }
    print(f'BlockCluster tt time {results["user"]+results["sys"]}')
    pickle.dump(
        results,
        open(
            results_folder
            + dataset_file.split("/")[-1].split(".")[0]
            + "_bc.pkl",
            "wb",
        ),
    )
    return results


for dataset_file in dataset_files:
    print("Start processing ", dataset_file.split("/")[-1])
    dataset = pickle.load(open(dataset_file, "rb"))

    graph = dataset["data"]
    row_cluster_indicator = dataset["row_cluster_indicator"]
    column_cluster_indicator = dataset["column_cluster_indicator"]
    row_clusters_index = row_cluster_indicator.argmax(1)
    column_clusters_index = column_cluster_indicator.argmax(1)
    nb_row_clusters, nb_column_clusters = (
        row_cluster_indicator.shape[1],
        column_cluster_indicator.shape[1],
    )

    if use_sp:
        train_with_sparsebm(
            dataset_file,
            graph,
            nb_row_clusters,
            nb_column_clusters,
            row_clusters_index,
            column_clusters_index,
            use_gpu,
        )
    if use_bm:
        train_with_blockmodels(
            dataset_file,
            graph,
            nb_row_clusters,
            nb_column_clusters,
            row_clusters_index,
            column_clusters_index,
        )
    if use_bc:
        train_with_blockcluster(
            dataset_file,
            graph,
            nb_row_clusters,
            nb_column_clusters,
            row_clusters_index,
            column_clusters_index,
        )
