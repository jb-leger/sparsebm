import numpy as np
from matplotlib import rc

rc("text", usetex=True)
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
connection_probabilities = np.array(
    [[4 * e, e, e, e * 2], [e, e, e, e], [2 * e, e, 2 * e, 2 * e]]
)

for file in dataset_files:
    results = pickle.load(open(file, "rb"))
    n1 = results["model"]["tau_1"].shape[0]
    n2 = results["model"]["tau_2"].shape[0]
    time_results_sparse[(n1, n2)].append(results["end_time"])
    cari_results_sparse[(n1, n2)].append(results["co_ari"])
    if results["end_time_not_sparse"]:
        cari_results_not_sparse[(n1, n2)].append(results["co_ari_not_sparse"])
        time_results_not_sparse[(n1, n2)].append(
            results["end_time_not_sparse"]
        )


xs = sorted(list(time_results_sparse.keys()), key=lambda x: x[0])
# xs = xs[:7]

############################ PLOTTING bayes error and Classification error ########################

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.plot(
    [a[0] for a in xs],
    [np.median(time_results_sparse[x]) for x in xs],
    marker="^",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:green"],
)
bp = ax.boxplot(
    [time_results_sparse[x] for x in xs],
    positions=[a[0] for a in xs],
    showfliers=False,
    capprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    whiskerprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    boxprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    medianprops=dict(
        linestyle="-",
        linewidth=0.35,
        color=mcolors.TABLEAU_COLORS["tab:green"],
    ),
    widths=[100] * len(xs),
)

ax.plot(
    [a[0] for a in xs[:8]],
    [np.median(time_results_not_sparse[x]) for x in xs[:8]],
    marker="*",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:blue"],
)
# ax.errorbar(
#     [a[0] for a in xs[:8]],
#     [np.mean(time_results_not_sparse[x]) for x in xs[:8]],
#     xerr=0.5,
#     yerr=2 * np.array([np.std(time_results_not_sparse[x]) for x in xs[:8]]),
#     linestyle="",
# )

bp = ax.boxplot(
    [time_results_not_sparse[x] for x in xs[:8]],
    positions=[a[0] for a in xs[:8]],
    showfliers=False,
    capprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    whiskerprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    boxprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    medianprops=dict(
        linestyle="-", linewidth=0.35, color=mcolors.TABLEAU_COLORS["tab:blue"]
    ),
    widths=[100] * len(xs[:8]),
)

ax.set_ylim(bottom=0)
ax.set_ylabel("Execution time (sec.)", size=12)
ax.set_xlabel("Network size $(n_1, n_2)$", size=12)
x_ticks = np.concatenate((xs[:1], xs[6:]))
ax.set_xticks([a[0] for a in x_ticks])
ax.set_xticklabels([str(n1) + "\n" + str(n2) for (n1, n2) in x_ticks])
plt.show()
