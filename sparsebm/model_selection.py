import numpy as np
import matplotlib.pyplot as plt
from . import SBM, LBM
from .utils import (
    lbm_merge_group,
    sbm_merge_group,
    lbm_split_group,
    sbm_split_group,
)
from typing import Any, Tuple, Union, Optional
from scipy.sparse import spmatrix
import scipy.sparse as sp
import logging

logger = logging.getLogger(__name__)

try:
    import cupy
    import cupyx
    import GPUtil

    _CUPY_INSTALLED = True
    _DEFAULT_USE_GPU = True
except ImportError:
    _CUPY_INSTALLED = False
    _DEFAULT_USE_GPU = False


class ModelSelection:
    """
    Explore and select the optimal number of classes for the LBM or SBM model.
    The best model is chosen according to the Integrated Completed Likelihood.
    A strategy of merging and splitting classes to produce good initializations is used.

    Examples
    --------
    >>> lbm_model_selection = ModelSelection(
    ...     model_type="LBM",
    ...     plot=True,
    ... )
    >>> lbm_selected = lbm_model_selection.fit(graph)
    """

    def __init__(
        self,
        model_type: str,
        *,
        n_clusters_max: Optional[int] = 30,
        use_gpu: Optional[bool] = True,
        gpu_index: Optional[int] = None,
        plot: Optional[bool] = True,
    ) -> None:
        """
        Parameters
        ----------
        model_type : {'LBM', 'SBM'}
            The type of co-clustering model to use.
        n_clusters_max : int, optional, default: 30
            Upper limit of the number of classes.
        use_gpu : bool, optional, default: True
            Specify if a GPU should be used.
        gpu_index : int, optional, default: None
            Specify the gpu index if needed.
        plot : bool, optional, default: True
            Display model exploration plot.
        """
        if not (model_type == "LBM" or model_type == "SBM"):
            raise Exception("model_type parameter must be 'SBM' or 'LBM'")

        self._model_type = model_type
        self._use_gpu = use_gpu if (use_gpu and _CUPY_INSTALLED) else False
        self._gpu_index = gpu_index
        self._plot = plot
        self._figure = plt.subplots(1) if plot else None
        self.model_explored = None
        self.n_clusters_max = n_clusters_max

    @property
    def selected_model(self) -> Union[LBM, SBM]:
        """sparsebm.LBM or sparsebm.SBM: Returns the optimal model explore so far."""
        assert self.model_explored, "Model selection not trained. Use fit()"
        return max(
            [m["model"] for m in self.model_explored.values()],
            key=lambda x: x.get_ICL(),
        )

    def fit(
        self,
        graph: Union[spmatrix, np.ndarray],
        symmetric: Optional[bool] = False,
    ) -> Union[LBM, SBM]:
        """ Perform model selection of the co-clustering.

        Parameters
        ----------
        graph : numpy.ndarray or scipy.sparse.spmatrix, shape=(n_samples, n_features) for the LBM or (n_samples, n_samples) for the SBM
            Matrix to be analyzed
        symmetric : bool, optional, default: False
            In case of SBM model, specify if the graph connections are symmetric.

        Returns
        -------
        sparsebm.LBM or sparsebm.SBM
            The best trained model according to the ICL.
        """
        if self._model_type == "SBM" and graph.shape[0] != graph.shape[0]:
            raise Exception(
                "For SBM, graph shapes must be equals (n_samples, n_samples)."
            )

        self._symmetric = symmetric
        self.graph = graph
        self._indices_ones = np.asarray(list(graph.nonzero()))
        self._row_col_degrees = (
            np.asarray(graph.sum(1)).squeeze(),
            np.asarray(graph.sum(0)).squeeze(),
        )

        self._X = sp.csr_matrix(graph)
        if self._use_gpu:
            self._X = cupyx.scipy.sparse.csr_matrix(self._X.astype(float))

        # Instantiate and training first model.
        if self._model_type == "LBM":
            model = LBM(
                1,
                1,
                max_iter=5000,
                n_init=1,
                n_init_total_run=1,
                n_iter_early_stop=1,
                verbosity=0,
                use_gpu=self._use_gpu,
                gpu_index=self._gpu_index,
            )
            model.fit(graph)
        else:
            model = SBM(
                1,
                max_iter=5000,
                n_init=1,
                n_init_total_run=1,
                n_iter_early_stop=1,
                verbosity=0,
                use_gpu=self._use_gpu,
                gpu_index=self._gpu_index,
            )
            model.fit(graph, symmetric=symmetric)
        nnq = (
            model.n_row_clusters + model.n_column_clusters
            if self._model_type == "LBM"
            else model.n_clusters
        )
        self.model_explored = {
            nnq: {
                "split_explored": False,
                "merge_explored": True,
                "model": model,
                "icl": model.get_ICL(),
            }
        }

        best_icl = [self.selected_model.get_ICL()]
        try:
            while not np.all(
                [
                    [m["merge_explored"], m["split_explored"]]
                    for m in self.model_explored.values()
                ]
            ):
                logger.info("Spliting")
                self.model_explored = self._explore_strategy(strategy="split")
                logger.info("Merging")
                self.model_explored = self._explore_strategy(strategy="merge")
                best_iter_model = self.selected_model
                best_icl.append(best_iter_model.get_ICL())
                logger.info("Best icl is {:.4f}".format(best_icl[-1]))
                if len(best_icl) > 3 and best_icl[-3] == best_icl[-1]:
                    break
        except KeyboardInterrupt:
            pass

        if self._plot:
            figure, _ = self._figure
            plt.close(figure)
        return self.selected_model

    def __repr__(self) -> str:
        return f"""ModelSelection(
                    graph=<{type(self.graph).__name__} at {hex(id(self.graph))}>,
                    model_type={self._model_type},
                    use_gpu={self._use_gpu},
                    symmetric={self._symmetric},
                )"""

    def _explore_strategy(self, strategy: str):
        """ Perform a splitting or merging strategy.

        The splitting strategy stops when the number of classes is greater
        than  min(1.5*number of classes of the best model,
        number of classes of the best model + 10, number of classes max).
        The merging strategy stops when the minimum relevant number of
        classes is reached.

        Parameters
        ----------
        strategy : {'merge', 'split'}
            The type of strategy.

        Returns
        -------
        model_explored: dict of {int: dict}
            All the models explored by the strategy. Keys of model_explored is
            the number of classes. The values are dict containing the model,
            its ICL value, two flags merge_explored and split_explored.

        """
        assert strategy in ["merge", "split"]

        # Getting the first model to explore, different according to the strategy.
        pv_model = (  # Number of classes, different according to the model LBM/SBM.
            self.model_explored[max(self.model_explored.keys())]
            if strategy == "merge"
            else self.model_explored[min(self.model_explored.keys())]
        )
        nnq_best_model = (
            (
                pv_model["model"].n_row_clusters
                + pv_model["model"].n_column_clusters
            )
            if self._model_type == "LBM"
            else pv_model["model"].n_clusters
        )

        model_explored = {}  # All models explored for the current strategy.
        best_model = pv_model  # Best model of the current strategy.

        models_to_explore = [pv_model]

        while models_to_explore:
            model_flag = models_to_explore.pop(0)
            nnq = (  # Number of classes, different according to the model LBM/SBM.
                model_flag["model"].n_row_clusters
                + model_flag["model"].n_column_clusters
                if self._model_type == "LBM"
                else model_flag["model"].n_clusters
            )
            model_explored[nnq] = model_flag

            if self._plot:
                _plot_merge_split_graph(
                    self, model_explored, strategy, best_model
                )

            flag_key = (
                "merge_explored" if strategy == "merge" else "split_explored"
            )
            classes_key = (nnq - 1) if strategy == "merge" else (nnq + 1)
            if model_flag[flag_key]:
                if classes_key in self.model_explored:
                    models_to_explore.append(self.model_explored[classes_key])
                    if (
                        self.model_explored[classes_key]["icl"]
                        > best_model["icl"]
                    ):
                        best_model = self.model_explored[classes_key]
                        nnq_best_model = (
                            (
                                best_model["model"].n_row_clusters
                                + best_model["model"].n_column_clusters
                            )
                            if self._model_type == "LBM"
                            else best_model["model"].n_clusters
                        )

                    logger.info(
                        "\t Already explored models from {} classes".format(
                            nnq
                        )
                    )
                    continue
            model_flag[flag_key] = True
            logger.info("\t Explore models from {} classes".format(nnq))

            if self._model_type == "LBM":
                # Explore all models derived from the strategy on the rows.
                r_icl, r_model = self._select_and_train_best_model(
                    model_flag["model"], strategy=strategy, type=0  # rows
                )
                # Explore all models derived from the strategy on the columns.
                c_icl, c_model = self._select_and_train_best_model(
                    model_flag["model"], strategy=strategy, type=1  # columns
                )
            else:
                r_icl, r_model = self._select_and_train_best_model(
                    model_flag["model"], strategy=strategy
                )
                c_icl, c_model = (-np.inf, None)

            best_models = [
                {
                    "model": r_model,
                    "merge_explored": False,
                    "split_explored": False,
                    "icl": r_icl,
                },
                {
                    "model": c_model,
                    "merge_explored": False,
                    "split_explored": False,
                    "icl": c_icl,
                },
            ]

            # Adding the model from previous strategy.
            if classes_key in self.model_explored:
                best_models = [self.model_explored[classes_key]] + best_models

            best_models.sort(key=lambda x: x["icl"], reverse=True)
            best_models = [d for d in best_models if not np.isinf(d["icl"])]
            if best_models:
                bfm = best_models[0]
                nnq_bm = (
                    bfm["model"].n_row_clusters
                    + bfm["model"].n_column_clusters
                    if self._model_type == "LBM"
                    else bfm["model"].n_clusters
                )

                if bfm["icl"] > best_model["icl"]:
                    best_model = bfm
                    nnq_best_model = (
                        (
                            best_model["model"].n_row_clusters
                            + best_model["model"].n_column_clusters
                        )
                        if self._model_type == "LBM"
                        else best_model["model"].n_clusters
                    )

                if strategy == "split" and (
                    (nnq_bm)
                    < min(
                        1.5 * (nnq_best_model),
                        nnq_best_model + 10,
                        self.n_clusters_max,
                    )
                    or nnq_bm < 4
                ):
                    models_to_explore.append(bfm)
                elif strategy == "split":
                    bfm["split_explored"] = True
                    model_explored[nnq_bm] = bfm

                if strategy == "merge" and (nnq_bm) > 3:
                    models_to_explore.append(bfm)
                elif strategy == "merge":
                    bfm["merge_explored"] = True
                    model_explored[nnq_bm] = bfm

        return model_explored

    def _select_and_train_best_model(
        self, model: Union[LBM, SBM], strategy: str, type: int = None
    ) -> Tuple[float, Union[LBM, SBM]]:
        """ Given model and a strategy, perform all possible merges/splits of
        classes and return the best one.

        The algorithm instantiate all merges/splits possible, n best models are
        selected and trained for a few steps and the best of them is trained until
        convergence.

        Parameters
        ----------
        model : sparsebm.LBM or sparsebm.SBM
            The model from which all merges/splits are tested.
        strategy : {'merge', 'split'}
            The type of strategy.

        type : int, optional
            0 for rows merging/splitting, 1 for columns merging/splitting

        Returns
        -------
        tuple of (float, sparsebm.LBM or sparsebm.SBM)
            The higher ICL value and its associated model, from all merges/splits.
        """
        assert strategy in ["merge", "split"]

        if self._model_type == "LBM":
            assert type in [0, 1]
            nb_clusters = (
                model.n_row_clusters if type == 0 else model.n_column_clusters
            )
            if strategy == "merge" and (
                (type == 0 and model.n_row_clusters <= 1)
                or (type == 1 and model.n_column_clusters <= 1)
            ):
                return (-np.inf, None)

        else:
            nb_clusters = model.n_clusters
            if strategy == "merge" and nb_clusters <= 1:
                return (-np.inf, None)

        # Getting all possible models from merge or split.
        if strategy == "merge":
            if self._model_type == "LBM":
                models = [
                    lbm_merge_group(
                        model.copy(),
                        type=type,
                        idx_group_1=a,
                        idx_group_2=b,
                        indices_ones=self._indices_ones,
                    )
                    for b in range(nb_clusters)
                    for a in range(b)
                ]
            else:
                models = [
                    sbm_merge_group(
                        model.copy(),
                        idx_group_1=a,
                        idx_group_2=b,
                        indices_ones=self._indices_ones,
                    )
                    for b in range(nb_clusters)
                    for a in range(b)
                ]
        else:
            if self._model_type == "LBM":
                models = [
                    lbm_split_group(
                        model.copy(),
                        self._row_col_degrees,
                        type=type,
                        index=i,
                        indices_ones=self._indices_ones,
                    )
                    for i in range(nb_clusters)
                ]
            else:
                models = [
                    sbm_split_group(
                        model.copy(),
                        self._row_col_degrees[0],
                        index=i,
                        indices_ones=self._indices_ones,
                    )
                    for i in range(nb_clusters)
                ]

        models.sort(key=lambda x: x[0], reverse=True)
        models = [(ic, m) for ic, m in models if not np.isinf(ic)]
        if not models:
            return (-np.inf, None)

        # Five best models are selected and trained for a few EM steps.
        for ic, m in models[:5]:
            if self._model_type == "LBM":
                m._fit_single(
                    self._X,
                    self._indices_ones,
                    self.graph.shape[0],
                    self.graph.shape[1],
                    init_params=True,
                    in_place=True,
                    early_stop=15,
                )
            else:
                m._fit_single(
                    self._X,
                    self._indices_ones,
                    self.graph.shape[0],
                    init_params=True,
                    in_place=True,
                    early_stop=15,
                )
        models = [(m.get_ICL(), m) for _, m in models[:5]]
        models.sort(key=lambda x: x[0], reverse=True)

        # The best model is trained until convergence.
        if self._model_type == "LBM":
            models[0][1]._fit_single(
                self._X,
                self._indices_ones,
                self.graph.shape[0],
                self.graph.shape[1],
                init_params=True,
                in_place=True,
            )
        else:
            models[0][1]._fit_single(
                self._X,
                self._indices_ones,
                self.graph.shape[0],
                init_params=True,
                in_place=True,
            )

        return (models[0][1].get_ICL(), models[0][1])


def _plot_merge_split_graph(
    model_selection, model_explored, strategy, best_model_current_strategy
):
    # figure = plt.figure(figsize=(5, 1))
    figure, ax = model_selection._figure
    ax.cla()
    if model_selection._model_type == "LBM":

        currently_explored_model = (
            model_explored[min(model_explored.keys())]["model"]
            if strategy == "merge"
            else model_explored[max(model_explored.keys())]["model"]
        )
        currently_explored_nqnl = (
            [
                (
                    currently_explored_model.n_row_clusters - 1,
                    currently_explored_model.n_column_clusters,
                ),
                (
                    currently_explored_model.n_row_clusters,
                    currently_explored_model.n_column_clusters - 1,
                ),
            ]
            if strategy == "merge"
            else [
                (
                    currently_explored_model.n_row_clusters + 1,
                    currently_explored_model.n_column_clusters,
                ),
                (
                    currently_explored_model.n_row_clusters,
                    currently_explored_model.n_column_clusters + 1,
                ),
            ]
        )
        nqs = [m["model"].n_row_clusters for m in model_explored.values()]
        nls = [m["model"].n_column_clusters for m in model_explored.values()]
        nqs_prev = [
            m["model"].n_row_clusters
            for m in model_selection.model_explored.values()
        ]
        nls_prev = [
            m["model"].n_column_clusters
            for m in model_selection.model_explored.values()
        ]

        ax.set_xlim((0, max(10, max(nqs), max(nqs_prev))))
        ax.set_ylim((0, max(10, max(nls), max(nls_prev))))
        if strategy == "merge":
            ax.set_title("Merging step")
        else:
            ax.set_title("Spliting step")
        ax.set_ylabel("Number of column groups")
        ax.set_xlabel("Number of row groups")
        ax.grid()

        ax.scatter(
            nqs_prev,
            nls_prev,
            s=100,
            c="grey",
            marker="+",
            label="Models explored during previous step",
        )
        ax.scatter(
            nqs,
            nls,
            s=70,
            c="orange",
            marker="o",
            label="Models explored at current step",
        )
        ax.scatter(
            [model_selection.selected_model.n_row_clusters],
            [model_selection.selected_model.n_column_clusters],
            s=120,
            c="black",
            marker="*",
            label="Current optimal model",
        )
        ax.annotate(
            str(round(model_selection.selected_model.get_ICL(), 2)),
            xy=(
                model_selection.selected_model.n_row_clusters - 0.5,
                model_selection.selected_model.n_column_clusters + 0.25,
            ),
        )
    else:
        nqs = [m["model"].n_clusters for m in model_explored.values()]
        icls = [m["model"].get_ICL() for m in model_explored.values()]
        nqs_prev = [
            m["model"].n_clusters
            for m in model_selection.model_explored.values()
        ]
        icls_prev = [
            m["model"].get_ICL()
            for m in model_selection.model_explored.values()
        ]

        ax.set_xlim((0, max(10, max(nqs), max(nqs_prev))))
        if strategy == "merge":
            ax.set_title("Merging step")
        else:
            ax.set_title("Spliting step")
        ax.set_ylabel("ICL")
        ax.set_xlabel("Number of row groups")
        ax.grid()
        ax.scatter(
            nqs_prev,
            icls_prev,
            s=100,
            c="grey",
            marker="+",
            label="Models explored during previous step",
        )
        ax.scatter(
            nqs,
            icls,
            s=70,
            c="orange",
            marker="o",
            label="Models explored at current step",
        )
        ax.scatter(
            [model_selection.selected_model.n_clusters],
            [model_selection.selected_model.get_ICL()],
            s=120,
            c="black",
            marker="*",
            label="Current optimal model",
        )
    ax.legend()
    plt.pause(0.01)
