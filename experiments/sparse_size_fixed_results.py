import numpy as np
from matplotlib import rc

# rc("text", usetex=True)
import matplotlib.pyplot as plt
import glob
import pickle
import time
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick

dataset_files = glob.glob("./experiments/results/size_fixed/*.pkl")

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

############################ PLOTTING bayes error and Classification error ########################
def epsilon_to_rate(x):
    return 1 - (connection_probabilities).mean() / (2 ** x)


def rate_to_epsilon(x):
    eps = 1e-10
    x2 = x.copy()
    x2[x2 >= 1] = 1 - eps
    results = -(
        np.log(1 - x2) - np.log((connection_probabilities).mean())
    ) / np.log(2)
    results[x >= 1] = xs.max()
    return results


fig, ax = plt.subplots(1, 1, figsize=(7, 4))
xs_values = epsilon_to_rate(xs)

ax.plot(
    xs_values,
    [np.median(time_results_sparse[x]) for x in xs],
    marker="^",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:green"],
)
bp = ax.boxplot(
    [time_results_sparse[x] for x in xs],
    positions=xs_values,
    showfliers=False,
    capprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    whiskerprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    boxprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    medianprops=dict(
        linestyle="-",
        linewidth=0.35,
        color=mcolors.TABLEAU_COLORS["tab:green"],
    ),
    widths=[0.005] * len(xs),
)

ax.plot(
    xs_values,
    [np.median(time_results_not_sparse[x]) for x in xs],
    marker="*",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:blue"],
)
bp = ax.boxplot(
    [time_results_not_sparse[x] for x in xs],
    positions=xs_values,
    showfliers=False,
    capprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    whiskerprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    boxprops=dict(linestyle="-", linewidth=0.35, color="grey"),
    medianprops=dict(
        linestyle="-", linewidth=0.35, color=mcolors.TABLEAU_COLORS["tab:blue"]
    ),
    widths=[0.005] * len(xs),
)

ax.set_ylabel("Execution time (sec.)", size=12)

ax.set_xlabel("sparsity rate", size=12)
ax.set_xlim(xs_values.min() - 0.01, 1)
ax.set_xticks(xs_values[:-2])  # uncomment if display issues.
ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
secax = ax.secondary_xaxis("top", functions=(rate_to_epsilon, epsilon_to_rate))
secax.set_xlabel("$\epsilon$")
secax.set_xticks(rate_to_epsilon(xs_values))

plt.show()
fig.savefig("experiments/results/size_fixed.png")
print("Figure saved in " + "experiments/results/size_fixed.png")
