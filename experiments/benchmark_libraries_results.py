from matplotlib import rc

# rc("text", usetex=True)
import pickle
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sp_lib = defaultdict(list)
sp_lib_gpu = defaultdict(list)
bm_lib = defaultdict(list)
bc_lib = defaultdict(list)
files = glob.glob(
    "./experiments/results/benchmark_libraries/size_growing/*.pkl"
)
for file in files:
    r = pickle.load(open(file, "rb"))
    if r["lib"] == "sparsebm":
        if r["gpu"]:
            sp_lib_gpu[(r["n1"], r["n2"])].append(r)
        else:
            sp_lib[(r["n1"], r["n2"])].append(r)
    if r["lib"] == "blockcluster":
        bc_lib[(r["n1"], r["n2"])].append(r)
    if r["lib"] == "blockmodels":
        bm_lib[(r["n1"], r["n2"])].append(r)

res_sp = np.array(
    [
        [
            k[0],
            k[1],
            np.median([v["sys"] + v["user"] for v in vs]),
            np.median([v["cari"] for v in vs]),
        ]
        for k, vs in sp_lib.items()
    ]
)
res_sp = np.sort(res_sp, 0)

# res_sp_gpu = np.array(
#     [
#         [
#             k[0],
#             k[1],
#             np.median([v["real"] for v in vs]),
#             np.median([v["cari"] for v in vs]),
#         ]
#         for k, vs in sp_lib_gpu.items()
#     ]
# )
# res_sp_gpu = np.sort(res_sp_gpu, 0)

bm_res = np.array(
    [
        [
            k[0],
            k[1],
            np.median([v["sys"] + v["user"] for v in vs]),
            np.median([v["cari"] for v in vs]),
        ]
        for k, vs in bm_lib.items()
    ]
)
bm_res = np.sort(bm_res, 0)

bc_res = np.array(
    [
        [
            k[0],
            k[1],
            np.median([v["sys"] + v["user"] for v in vs]),
            np.median([v["cari"] for v in vs]),
        ]
        for k, vs in bc_lib.items()
    ]
)
bc_res = np.sort(bc_res, 0)
import matplotlib

font = {"size": 14}

matplotlib.rc("font", **font)
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.plot(
    res_sp[:, 0] * res_sp[:, 1],
    res_sp[:, 2],
    marker="^",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:green"],
    label="SparseBM CPU time",
)
# ax.plot(
#     res_sp_gpu[:, 0] * res_sp_gpu[:, 1],
#     res_sp_gpu[:, 2],
#     marker=".",
#     markersize=7,
#     linewidth=0.5,
#     color=mcolors.TABLEAU_COLORS["tab:green"],
#     label="SparseBM GPU effective time",
# )
ax.plot(
    bm_res[:, 0] * bm_res[:, 1],
    bm_res[:, 2],
    marker="*",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:blue"],
    label="Blockmodels CPU time",
)
# ax.annotate(
#     "OOM",
#     (-0.05 * 10 ** 8 + bm_res[-1, 0] * bm_res[-1, 1], 80 + bm_res[-1, 2]),
#     color=mcolors.TABLEAU_COLORS["tab:blue"],
# )
ax.plot(
    bc_res[:, 0] * bc_res[:, 1],
    bc_res[:, 2],
    marker="+",
    markersize=7,
    linewidth=0.5,
    color=mcolors.TABLEAU_COLORS["tab:red"],
    label="Blockcluster CPU time",
)
# ax.annotate(
#     "OOM",
#     (-0.05 * 10 ** 8 + bc_res[-1, 0] * bc_res[-1, 1], 50 + bc_res[-1, 2]),
#     color=mcolors.TABLEAU_COLORS["tab:red"],
# )

ax.set_ylabel("Time in seconds")
ax.set_xlabel("Network size $(n_1 \cdot n_2)$")
ax.ticklabel_format(style="sci", axis="x")
plt.legend()
plt.show()
