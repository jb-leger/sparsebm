import numpy as np
import matplotlib.pyplot as plt
from . import SBM_bernouilli, LBM_bernouilli
from .utils import (
    lbm_merge_group,
    sbm_merge_group,
    lbm_split_group,
    sbm_split_group,
)
from typing import Any, Tuple, Union, Optional
from scipy.sparse import spmatrix


class ModelSelection:
    """
    Explore and select the optimal number of classes for the LBM or SBM model.
    The best model is chosen according to the Integrated Completed Likelihood.
    A strategy of merging and splitting classes to produce good initializations is used.
    """

    def __init__(
        self,
        graph: Union[spmatrix, np.ndarray],
        model_type: str,
        gpu_number: Optional[int] = 0,
        symetric: Optional[bool] = False,
    ) -> None:
        """
        Parameters
        ----------
        graph : numpy.ndarray or scipy.sparse.spmatrix, shape=(n_samples, n_features) for the LBM or (n_samples, n_samples) for the SBM
            Matrix to be analyzed
        model_type : str
            Either "LBM" or "SBM" the type of co-clustering model to use.
        gpu_number : int, optional, default: 0
            Select the index of the GPU. None if no need of GPU.
        symetric : bool, optional, default: False
            In case of SBM model, specify if the graph connections are symetric.
        """
        if not (model_type == "LBM" or model_type == "SBM"):
            raise Exception("model_type parameter must be 'SBM' or 'LBM'")
        if model_type == "SBM" and graph.shape[0] != graph.shape[0]:
            raise Exception(
                "For SBM, graph shapes must be equals (n_samples, n_samples)."
            )

        self.graph = graph
        self._indices_ones = np.asarray(list(graph.nonzero()))
        self._row_col_degrees = (
            np.asarray(graph.sum(1)).squeeze(),
            np.asarray(graph.sum(0)).squeeze(),
        )
        self._model_type = model_type
        self._gpu_number = gpu_number
        self._symetric = symetric

        if model_type == "LBM":
            model = LBM_bernouilli(
                1,
                1,
                max_iter=5000,
                n_init=1,
                n_init_total_run=1,
                n_iter_early_stop=1,
                tol=1e-3,
                verbosity=0,
                gpu_number=gpu_number,
            )
        else:
            model = SBM_bernouilli(
                1,
                gpu_number=gpu_number,
                symetric=symetric,
                max_iter=5000,
                n_init=1,
                n_init_total_run=1,
                n_iter_early_stop=1,
                tol=1e-3,
                verbosity=0,
            )
        model.fit(graph)

        nnq = (
            model._n_row_clusters + model._n_column_clusters
            if self._model_type == "LBM"
            else model._n_clusters
        )
        self.model_explored = {
            nnq: {
                "split_explored": False,
                "merge_explored": True,
                "model": model,
                "icl": model.get_ICL(),
            }
        }

    @property
    def selected_model(self) -> Union[LBM_bernouilli, SBM_bernouilli]:
        """sparsebm.LBM_bernouilli or sparsebm.SBM_bernouilli: Returns the optimal model explore so far."""
        return max(
            [m["model"] for m in self.model_explored.values()],
            key=lambda x: x.get_ICL(),
        )

    def fit(self) -> Union[LBM_bernouilli, SBM_bernouilli]:
        """ Perform model selection of the co-clustering.

        Returns
        -------
        sparsebm.LBM_bernouilli or sparsebm.SBM_bernouilli
            The best trained model according to the ICL.
        """
        try:
            while not np.all(
                [
                    [m["merge_explored"], m["split_explored"]]
                    for m in self.model_explored.values()
                ]
            ):
                print("Spliting")
                self.model_explored = self._explore_strategy(strategy="split")
                print("Merging")
                self.model_explored = self._explore_strategy(strategy="merge")

        except KeyboardInterrupt:
            pass

        return self.selected_model

    def __repr__(self) -> str:
        return f"""ModelSelection(
                    graph=graph,
                    model_type={self._model_type},
                    gpu_number={self._gpu_number},
                    symetric={self._symetric},
                )"""

    def _explore_strategy(self, strategy: str):
        """ Perform a splitting or merging strategy.

        The splitting strategy stops when the number of classes is greater
        than  min(1.5*number of classes of the best model,
        number of classes of the best model + 10).
        The merging strategy stops when the minimum relevant number of
        classes is reached.

        Parameters
        ----------
        strategy : str
            The type of strategy. Either 'merge' or 'split'

        Returns
        -------
        model_explored: dict of {int: dict}
            All the models explored by the strategy. Keys of model_explored is
            the number of classes. The values are dict containing the model,
            its ICL value, two flags merge_explored and split_explored.

        """
        assert strategy == "merge" or strategy == "split"

        # Getting the first model to explore, different according to the strategy.
        pv_model = (  # Number of classes, different according to the model LBM/SBM.
            self.model_explored[max(self.model_explored.keys())]
            if strategy == "merge"
            else self.model_explored[min(self.model_explored.keys())]
        )
        nnq_best_model = (
            (
                pv_model["model"]._n_row_clusters
                + pv_model["model"]._n_column_clusters
            )
            if self._model_type == "LBM"
            else pv_model["model"]._n_clusters
        )

        model_explored = {}  # All models explored for the current strategy.
        best_model = pv_model  # Best model of the current strategy.

        models_to_explore = [pv_model]

        while models_to_explore:
            model_flag = models_to_explore.pop(0)
            nnq = (  # Number of classes, different according to the model LBM/SBM.
                model_flag["model"]._n_row_clusters
                + model_flag["model"]._n_column_clusters
                if self._model_type == "LBM"
                else model_flag["model"]._n_clusters
            )
            model_explored[nnq] = model_flag

            plot_merge_split_graph(
                model_explored, strategy, self.model_explored
            )

            flag_key = (
                "merge_explored" if strategy == "merge" else "split_explored"
            )
            classes_key = (nnq - 1) if strategy == "merge" else (nnq + 1)
            if model_flag[flag_key]:
                if classes_key in self.model_explored:
                    models_to_explore.append(self.model_explored[classes_key])
                    if (
                        icl_model.model_explored[classes_key]["icl"]
                        > best_model["icl"]
                    ):
                        best_model = icl_model.model_explored[classes_key]
                        nnq_best_model = (
                            (
                                best_model["model"]._n_row_clusters
                                + best_model["model"]._n_column_clusters
                            )
                            if self._model_type == "LBM"
                            else best_model["model"]._n_clusters
                        )

                    print(
                        "\t Already explored models from {} classes".format(
                            nnq
                        )
                    )
                    continue
            model_flag[flag_key] = True
            print("\t Explore models from {} classes".format(nnq))

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
                    bfm["model"]._n_row_clusters
                    + bfm["model"]._n_column_clusters
                    if self._model_type == "LBM"
                    else bfm["model"]._n_clusters
                )

                if bfm["icl"] > best_model["icl"]:
                    best_model = bfm
                    nnq_best_model = (
                        (
                            best_model["model"]._n_row_clusters
                            + best_model["model"]._n_column_clusters
                        )
                        if self._model_type == "LBM"
                        else best_model["model"]._n_clusters
                    )

                if strategy == "split" and (
                    (nnq_bm) < min(1.5 * (nnq_best_model), nnq_best_model + 10)
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
        self,
        model: Union[LBM_bernouilli, SBM_bernouilli],
        strategy: str,
        type: int = None,
    ) -> Tuple[float, Union[LBM_bernouilli, SBM_bernouilli]]:
        """ Given model and a strategy, perform all possible merges/splits of
        classes and return the best one.

        The algorithm instantiate all merges/splits possible, n best models are
        selected and trained for a few steps and the best of them is trained until
        convergence.

        Parameters
        ----------
        model : sparsebm.LBM_bernouilli or sparsebm.SBM_bernouilli
            The model from which all merges/splits are tested.
        strategy : str
            The type of strategy. Either 'merge' or 'split'

        type : int, optional
            0 for rows merging/splitting, 1 for columns merging/splitting

        Returns
        -------
        tuple of (float, sparsebm.LBM_bernouilli or sparsebm.SBM_bernouilli)
            The higher ICL value and its associated model, from all merges/splits.
        """
        assert strategy == "merge" or strategy == "split"

        if self._model_type == "LBM":
            assert type == 0 or type == 1
            nb_clusters = (
                model._n_row_clusters
                if type == 0
                else model._n_column_clusters
            )
            if strategy == "merge" and (
                (type == 0 and model._n_row_clusters <= 1)
                or (type == 1 and model._n_column_clusters <= 1)
            ):
                return (-np.inf, None)

        else:
            nb_clusters = model._n_clusters
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
                    for a in range(nb_clusters)
                    for b in range(nb_clusters)
                    if (a != b and a < b)
                ]
            else:
                models = [
                    sbm_merge_group(
                        model.copy(),
                        idx_group_1=a,
                        idx_group_2=b,
                        indices_ones=self._indices_ones,
                    )
                    for a in range(nb_clusters)
                    for b in range(nb_clusters)
                    if (a != b and a < b)
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
                    self._indices_ones,
                    self.graph.shape[0],
                    self.graph.shape[1],
                    init_params=True,
                    in_place=True,
                    early_stop=15,
                )
            else:
                m._fit_single(
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
                self._indices_ones,
                self.graph.shape[0],
                self.graph.shape[1],
                init_params=True,
                in_place=True,
            )
        else:
            models[0][1]._fit_single(
                self._indices_ones,
                self.graph.shape[0],
                init_params=True,
                in_place=True,
            )

        return (models[0][1].get_ICL(), models[0][1])


def plot_merge_split_graph(
    model_explored, strategy, previously_explored_models
):
    if isinstance(list(model_explored.values())[0]["model"], LBM_bernouilli):
        plt.cla()
        nqs = [m["model"]._n_row_clusters for m in model_explored.values()]
        nls = [m["model"]._n_column_clusters for m in model_explored.values()]
        nqs_prev = [
            m["model"]._n_row_clusters
            for m in previously_explored_models.values()
        ]
        nls_prev = [
            m["model"]._n_column_clusters
            for m in previously_explored_models.values()
        ]

        plt.xlim((0, max(10, max(nqs), max(nqs_prev))))
        plt.ylim((0, max(10, max(nls), max(nls_prev))))
        if strategy == "merge":
            plt.title("Merging strategy")
        else:
            plt.title("Spliting strategy")
        plt.ylabel("Number of column groups")
        plt.xlabel("Number of row groups")
        plt.grid()
        plt.scatter(
            nqs, nls, s=70, c="blue", marker="o", label="Current strategy path"
        )
        plt.scatter(
            nqs_prev,
            nls_prev,
            s=100,
            c="grey",
            marker="+",
            label="Previously strategy path",
        )
    else:
        plt.cla()
        nqs = [m["model"]._n_clusters for m in model_explored.values()]
        icls = [m["model"].get_ICL() for m in model_explored.values()]
        nqs_prev = [
            m["model"]._n_clusters for m in previously_explored_models.values()
        ]
        icls_prev = [
            m["model"].get_ICL() for m in previously_explored_models.values()
        ]
        plt.xlim((0, max(10, max(nqs), max(nqs_prev))))
        if strategy == "merge":
            plt.title("Merging strategy")
        else:
            plt.title("Spliting strategy")
        plt.ylabel("ICL")
        plt.xlabel("Number of row groups")
        plt.grid()
        plt.scatter(
            nqs,
            icls,
            s=70,
            c="blue",
            marker="o",
            label="Current strategy path",
        )
        plt.scatter(
            nqs_prev,
            icls_prev,
            s=100,
            c="grey",
            marker="+",
            label="Previously strategy path",
        )
    plt.legend()
    plt.pause(0.05)
