import numpy as np
from matplotlib import rc

rc("text", usetex=True)
import matplotlib.pyplot as plt
import glob
import pickle
import time
import matplotlib.colors as mcolors

dataset_files = glob.glob("./experiments/results/sparsity/*.pkl")

from collections import defaultdict

time_results_sparse = defaultdict(list)
time_results_not_sparse = defaultdict(list)


e = 0.25
connection_probabilities = np.array(
    [[4 * e, e, e, e * 2], [e, e, e, e], [2 * e, e, 2 * e, 2 * e]]
)

for file in dataset_files:
    results = pickle.load(open(file, "rb"))
    time_results_sparse[results["exponent"]].append(results["end_time"])
    time_results_not_sparse[results["exponent"]].append(
        results["end_time_not_sparse"]
    )

xs = np.sort(np.array(list(time_results_sparse.keys())))
xs = xs[:7]

############################ PLOTTING bayes error and Classification error ########################

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.plot(
    xs,
    [np.median(time_results_sparse[x]) for x in xs],
    marker="^",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:green"],
)
bp = ax.boxplot(
    [time_results_sparse[x] for x in xs],
    positions=xs,
    showfliers=False,
    capprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    whiskerprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    boxprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    medianprops=dict(
        linestyle="-",
        linewidth=0.35,
        color=mcolors.TABLEAU_COLORS["tab:green"],
    ),
    widths=[0.2] * len(xs),
)

ax.plot(
    xs,
    [np.median(time_results_not_sparse[x]) for x in xs],
    marker="*",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:blue"],
)
bp = ax.boxplot(
    [time_results_not_sparse[x] for x in xs],
    positions=xs,
    showfliers=False,
    capprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    whiskerprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    boxprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    medianprops=dict(
        linestyle="-", linewidth=0.35, color=mcolors.TABLEAU_COLORS["tab:blue"]
    ),
    widths=[0.2] * len(xs),
)

ax.set_ylim(bottom=0)
ax.set_ylabel("Execution time (sec.)", size=12)
ax.set_xlabel("$\epsilon$", size=12)
# ax.set_xticks([0,1,2,3,4])
# ax.set_xticklabels(["0", "1\n0.2","2\n0.1", "3", "4"])


def epsilon_to_rate(x):
    return (connection_probabilities).mean() / (2 ** x)


def rate_to_epsilon(x):
    eps = 1e-50
    x2 = x.copy()
    x2[x2 == 0] = eps
    results = -(
        np.log(x2) - np.log((connection_probabilities).mean())
    ) / np.log(2)
    results[x == 0] = 0
    return results


secax = ax.secondary_xaxis("top", functions=(epsilon_to_rate, rate_to_epsilon))
secax.set_xlabel("sparsity rate")
secax.set_xticks(epsilon_to_rate(xs))
plt.show()


# secax = ax.secondary_xaxis('top', functions=(forward, inverse))
# secax.set_xlabel('period [s]')
