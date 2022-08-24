from matplotlib import rc

# rc("text", usetex=True)
import matplotlib

# font = {"size": 14}
# matplotlib.rc("font", **font)


import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import time
import matplotlib.colors as mcolors

dataset_files = glob.glob("./experiments/results/sparsity_fixed/*.pkl")

from collections import defaultdict

time_results_sparse = defaultdict(list)
time_results_not_sparse = defaultdict(list)
cari_results_sparse = defaultdict(list)
cari_results_not_sparse = defaultdict(list)

e = 0.25
exponent = 5
connection_probabilities = (
    np.array([[4 * e, e, e, e * 2], [e, e, e, e], [2 * e, e, 2 * e, 2 * e]])
    / 2**exponent
)


for file in dataset_files:
    results = pickle.load(open(file, "rb"))
    n1 = results["model"]["tau_1"].shape[0]
    n2 = results["model"]["tau_2"].shape[0]
    time_results_sparse[(n1, n2)].append(results["end_time"])
    cari_results_sparse[(n1, n2)].append(results["co_ari"])
    if results["end_time_not_sparse"]:
        cari_results_not_sparse[(n1, n2)].append(results["co_ari_not_sparse"])
        time_results_not_sparse[(n1, n2)].append(results["end_time_not_sparse"])


xs = sorted(list(time_results_sparse.keys()), key=lambda x: x[0])

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
xs_values = [a * a / 2 for a in np.array([a[0] for a in xs])]
ax.plot(
    xs_values,
    [np.median(time_results_sparse[x]) for x in xs],
    marker="^",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:green"],
)
xs_value_not_sparse = [
    a * a / 2
    for a in np.array([a[0] for a in sorted(list(time_results_not_sparse.keys()))])
]
ax.plot(
    xs_value_not_sparse,
    [
        np.median(time_results_not_sparse[x])
        for x in sorted(list(time_results_not_sparse.keys()))
    ],
    marker="*",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:blue"],
)
# ax.annotate(
#     "OOM",
#     (
#         xs_value_not_sparse[-1],
#         20
#         + np.median(
#             time_results_not_sparse[
#                 sorted(list(time_results_not_sparse.keys()))[-1]
#             ]
#         ),
#     ),
#     color=mcolors.TABLEAU_COLORS["tab:blue"],
# )
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Execution time (sec.)")
ax.set_xlabel("Network size $(n_1 \cdot n_2)$")
# ax.ticklabel_format(style="sci", axis="x")
plt.show()
fig.savefig("experiments/results/sparsity_fixed.png")
print("Figure saved in " + "experiments/results/sparsity_fixed.png")
