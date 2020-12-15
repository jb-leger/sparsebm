import sys
import copy
import progressbar
import numpy as np
import scipy.sparse as sp
from heapq import heappush, heappushpop
from itertools import count
from sklearn.base import BaseEstimator
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


class LBM(BaseEstimator):
    """
    LBM with distribution.
    The class implements the random initialisation strategy.

    Notes
    -----
    Convergence of the EM algorithm is declared when
    new_loglikelihood - old_loglikelihood <=
    (`atol` + `rtol` * abs(new_loglikelihood)). The convergence is checked
    every 10 EM steps.


    Examples
    --------
    >>> from sparsebm import LBM
    >>> model = LBM(
    ...     nb_row_clusters=4,
    ...     nb_column_clusters=4,
    ...     n_init=100,
    ...     n_iter_early_stop=10,
    ...     n_init_total_run=5,
    ...     verbosity=1,
    ... )
    >>> model.fit(graph)
    """

    def __init__(
        self,
        n_row_clusters=4,
        n_column_clusters=4,
        *,
        max_iter=10000,
        n_init=100,
        n_init_total_run=10,
        n_iter_early_stop=10,
        rtol=1e-10,
        atol=1e-4,
        verbosity=1,
        use_gpu=_DEFAULT_USE_GPU,
        gpu_index=None,
    ):
        """
        Parameters
        ----------
        n_row_clusters : int
            Number of row clusters to form
        n_column_clusters : int
            Number of row clusters to form
        max_iter : int, optional, default: 10000
            Maximum number of EM iterations
        n_init : int, optional, default: 100
            Number of initializations that will be run for n_iter_early_stop EM iterations.
        n_init_total_run : int, optional, default: 10
            Number of the n_init best initializations that will be run until convergence.
        n_iter_early_stop : int, optional, default: 100
            Number of EM iterations to used to run the n_init initializations.
        rtol : float, default: 1e-10
            The relative tolerance parameter (see Notes).
        atol : float, default: 1e-4
            The absolute tolerance parameter (see Notes).
        verbosity : int, optional, default: 1
            Degree of verbosity. Scale from 0 (no message displayed) to 3.
        use_gpu : bool, optional, default: _DEFAULT_USE_GPU
            Specify if a GPU should be used.
        gpu_index : int, optional, default: None
            Specify the gpu index if needed.
        """
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_init_total_run = (
            n_init_total_run if n_init > n_init_total_run else n_init
        )
        self.nb_iter_early_stop = n_iter_early_stop
        self.atol = atol
        self.rtol = rtol
        self.verbosity = verbosity
        self.n_row_clusters = n_row_clusters
        self.n_column_clusters = n_column_clusters
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index

    @property
    def group_connection_probabilities(self):
        """array_like: Returns the group connection probabilities"""
        assert (
            self.trained_successfully_ == True
        ), "Model not trained successfully"
        return self.pi_

    @property
    def row_group_membership_probability(self):
        """array_like: Returns the row group membership probabilities"""
        assert (
            self.trained_successfully_ == True
        ), "Model not trained successfully"
        return self.alpha_1_

    @property
    def column_group_membership_probability(self):
        """array_like: Returns the column group membership probabilities"""
        assert (
            self.trained_successfully_ == True
        ), "Model not trained successfully"
        return self.alpha_2_

    @property
    def row_labels(self):
        """array_like: Returns the row labels"""
        assert (
            self.trained_successfully_ == True
        ), "Model not trained successfully"
        return self.tau_1_.argmax(1)

    @property
    def column_labels(self):
        """array_like: Returns the column labels"""
        assert (
            self.trained_successfully_ == True
        ), "Model not trained successfully"
        return self.tau_2_.argmax(1)

    @property
    def row_predict_proba(self):
        """array_like: Returns the predicted row classes membership probabilities"""
        assert self.trained_successfully_ == True
        return self.tau_1_

    @property
    def column_predict_proba(self):
        """array_like: Returns the predicted column classes membership probabilities"""
        assert (
            self.trained_successfully_ == True
        ), "Model not trained successfully"
        return self.tau_2_

    @property
    def trained_successfully(self):
        """bool: Returns the predicted column classes membership probabilities"""
        return self.trained_successfully_

    def get_params(self, deep=True):
        return {
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "n_init_total_run": self.n_init_total_run,
            "n_iter_early_stop": self.n_iter_early_stop,
            "rtol": self.rtol,
            "atol": self.atol,
            "verbosity": self.verbosity,
            "n_row_clusters": self.n_row_clusters,
            "n_column_clusters": self.n_column_clusters,
            "use_gpu": self.use_gpu,
            "gpu_index": self.gpu_index,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _check_params(self):
        self._np = np
        self._cupyx = None
        self.loglikelihood_ = -np.inf
        self.trained_successfully_ = False

        if self.use_gpu and (
            not _CUPY_INSTALLED
            or not _DEFAULT_USE_GPU
            or not cupy.cuda.is_available()
        ):
            self.gpu_number = None
            self.use_gpu = False
            logger.warning(
                "GPU not used as cupy library seems not to be installed or CUDA is not available"
            )

        if (
            self.use_gpu
            and _CUPY_INSTALLED
            and _DEFAULT_USE_GPU
            and cupy.cuda.is_available()
        ):
            if self.gpu_index != None:
                cupy.cuda.Device(self.gpu_index).use()
                self._np = cupy
                self._cupyx = cupyx
            else:
                free_idx = GPUtil.getAvailable("memory", limit=10)
                if not free_idx:
                    self.use_gpu = False
                    logger.warning("GPU not used as no gpu is free")
                else:
                    self._np = cupy
                    self._cupyx = cupyx
                    gpu_number = free_idx[0]
                    cupy.cuda.Device(gpu_number).use()

    def score(self, X, y=None):
        if not hasattr(self, "loglikelihood_"):
            self.fit(X)
        return self.get_ICL()

    def get_ICL(self) -> float:
        """Computation of the ICL criteria that can be used for model selection.
        Returns
        -------
        float
            value of the ICL criteria.
        """
        assert (
            self.trained_successfully_ == True
        ), "Model not trained successfully"
        return (
            self.loglikelihood_
            - (self.n_row_clusters - 1) / 2 * np.log(self._nb_rows)
            - (self.n_column_clusters - 1) / 2 * np.log(self._nb_cols)
            - (self.n_column_clusters * self.n_row_clusters)
            / 2
            * np.log(self._nb_cols * self._nb_rows)
        )

    def fit(self, X, y=None) -> None:
        """Perform co-clustering by direct maximization of graph modularity.

        Parameters
        ----------
        X : numpy matrix or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        self._check_params()
        self.trained_successfully_ = False
        n1, n2 = X.shape
        self._nb_rows = n1
        self._nb_cols = n2
        X = sp.csr_matrix(X)
        if self.use_gpu:
            X = self._cupyx.scipy.sparse.csr_matrix(X.astype(float))
            X_coo = X.tocoo()
            indices_ones = [X_coo.row, X_coo.col]
        else:
            indices_ones = list(X.nonzero())

        try:
            # Initialize and start to run each for a while.

            if self.verbosity > 0:
                logger.info(
                    "---------- START RANDOM INITIALIZATIONS ---------- "
                )
                bar = progressbar.ProgressBar(
                    max_value=self.n_init,
                    widgets=[
                        progressbar.SimpleProgress(),
                        " Initializations: ",
                        " [",
                        progressbar.Percentage(),
                        " ] ",
                        progressbar.Bar(),
                        " [ ",
                        progressbar.Timer(),
                        " ] ",
                    ],
                    redirect_stdout=True,
                ).start()

            best_inits = []
            tiebreaker = count()
            for run_number in range(self.n_init):
                if self.verbosity > 0:
                    bar.update(run_number)
                (
                    success,
                    ll,
                    pi,
                    alpha_1,
                    alpha_2,
                    tau_1,
                    tau_2,
                ) = self._fit_single(
                    X,
                    indices_ones,
                    n1,
                    n2,
                    early_stop=self.nb_iter_early_stop,
                    run_number=run_number,
                )
                calculation_result = [
                    ll,
                    next(tiebreaker),
                    pi,
                    alpha_1,
                    alpha_2,
                    tau_1,
                    tau_2,
                ]
                if len(best_inits) < max(1, int(self.n_init_total_run)):
                    heappush(best_inits, calculation_result)
                else:
                    heappushpop(best_inits, calculation_result)
            if self.verbosity > 0:
                bar.finish()
                logger.info(
                    "---------- START TRAINING BEST INITIALIZATIONS ---------- "
                )
                bar = progressbar.ProgressBar(
                    max_value=len(best_inits),
                    widgets=[
                        progressbar.SimpleProgress(),
                        " Runs: ",
                        " [",
                        progressbar.Percentage(),
                        " ] ",
                        progressbar.Bar(),
                        " [ ",
                        progressbar.Timer(),
                        " ] ",
                    ],
                    redirect_stdout=True,
                ).start()
            # Repeat the whole EM algorithm with several initializations.
            for run_number, init in enumerate(best_inits):
                if self.verbosity > 0:
                    bar.update(run_number)

                (pi, alpha_1, alpha_2, tau_1, tau_2) = (
                    init[2],
                    init[3],
                    init[4],
                    init[5],
                    init[6],
                )
                (
                    success,
                    ll,
                    pi,
                    alpha_1,
                    alpha_2,
                    tau_1,
                    tau_2,
                ) = self._fit_single(
                    X,
                    indices_ones,
                    n1,
                    n2,
                    init_params=(pi, alpha_1, alpha_2, tau_1, tau_2),
                    run_number=run_number,
                )

                if success and ll > self.loglikelihood_:
                    self.loglikelihood_ = ll.get() if self.use_gpu else ll
                    self.trained_successfully_ = True
                    self.pi_ = pi.get() if self.use_gpu else pi
                    self.alpha_1_ = alpha_1.get() if self.use_gpu else alpha_1
                    self.alpha_2_ = alpha_2.get() if self.use_gpu else alpha_2
                    self.tau_1_ = tau_1.get() if self.use_gpu else tau_1
                    self.tau_2_ = tau_2.get() if self.use_gpu else tau_2
        except KeyboardInterrupt:
            pass
        finally:
            if self.verbosity > 0:
                bar.finish()
        return self

    def _fit_single(
        self,
        X,
        indices_ones,
        n1,
        n2,
        early_stop=None,
        init_params=None,
        in_place=False,
        run_number=None,
    ):
        """Perform one run of the LBM algorithm with one random initialization.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape=(n1, n2)
            Matrix to be analyzed
        indices_ones : Non zero indices of the data matrix.
        n1 : Number of rows in the data matrix.
        n2 : Number of columns in the data matrix.
        """
        old_ll = -self._np.inf
        success = False

        if init_params:
            if init_params is True:
                if (
                    self.pi_ is not None
                    and self.alpha_1_ is not None
                    and self.alpha_2_ is not None
                    and self.tau_1_ is not None
                    and self.tau_2_ is not None
                ):
                    alpha_1, alpha_2, tau_1, tau_2, pi = (
                        self._np.asarray(self.alpha_1_),
                        self._np.asarray(self.alpha_2_),
                        self._np.asarray(self.tau_1_),
                        self._np.asarray(self.tau_2_),
                        self._np.asarray(self.pi_),
                    )
                else:
                    assert False
            else:
                (pi, alpha_1, alpha_2, tau_1, tau_2) = init_params
        else:
            alpha_1, alpha_2, tau_1, tau_2, pi = self._init_LBM_random(
                n1, n2, self.n_row_clusters, self.n_column_clusters, X.nnz
            )

        # Repeat EM step until convergence.
        for iteration in range(self.max_iter):
            if early_stop and iteration >= early_stop:
                ll = self._compute_likelihood(
                    indices_ones, pi, alpha_1, alpha_2, tau_1, tau_2
                )
                break
            if iteration % 5 == 0:
                ll = self._compute_likelihood(
                    indices_ones, pi, alpha_1, alpha_2, tau_1, tau_2
                )
                if (ll - old_ll) < (self.atol + self.rtol * self._np.abs(ll)):
                    success = True
                    break

                log_txt = f"\t EM Iter: {iteration:5d}  \t  log-like:{ll.get() if self.use_gpu else ll:.4f} \t diff:{self._np.abs(old_ll - ll).get() if self.use_gpu else self._np.abs(old_ll - ll):.6f}"
                if self.verbosity > 1:
                    logger.info(log_txt)
                else:
                    logger.debug(log_txt)
                old_ll = ll
            pi, alpha_1, alpha_2, tau_1, tau_2 = self._step_EM(
                X, indices_ones, pi, alpha_1, alpha_2, tau_1, tau_2, n1, n2
            )
        else:
            success = True
        if self.verbosity > 1 and run_number:
            logger.info(
                f"Run {run_number:3d} / {self.n_init:3d} \t success : {success} \t log-like: {ll.get()  if self.use_gpu else ll:.4f} \t nb_iter: {iteration:5d}"
            )

        if in_place:
            self.loglikelihood_ = ll.get() if self.use_gpu else ll
            self.trained_successfully_ = True
            self.pi_ = pi.get() if self.use_gpu else pi
            self.alpha_1_ = alpha_1.get() if self.use_gpu else alpha_1
            self.alpha_2_ = alpha_2.get() if self.use_gpu else alpha_2
            self.tau_1_ = tau_1.get() if self.use_gpu else tau_1
            self.tau_2_ = tau_2.get() if self.use_gpu else tau_2

        return success, ll, pi, alpha_1, alpha_2, tau_1, tau_2

    def _step_EM(
        self, X, indices_ones, pi, alpha_1, alpha_2, tau_1, tau_2, n1, n2
    ):
        """Realize EM step. Update both variationnal and model parameters.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape=(n1, n2)
            Matrix to be analyzed
        indices_ones : Non zero indices of the data matrix.
        pi : Connection probability matrix between row and column groups.
        alpha_1 : Row group model parameters.
        alpha_2 : Column group model parameters.
        tau_1 : Row group variationnal parameters.
        tau_2 : Column group variationnal parameters.
        n1 : Number of rows in the data matrix.
        n2 : Number of columns in the data matrix.
        """

        eps_1 = max(1e-4 / n1, 1e-9)
        eps_2 = max(1e-4 / n2, 1e-9)
        nq, nl = self.n_row_clusters, self.n_column_clusters

        ########################## E-step  ##########################
        u = X.dot(tau_2)  # Shape is (n1,nl)
        v = X.T.dot(tau_1)  # Shape is (n2,nq)

        # Update of tau_1 with sparsity trick.
        l_tau_1 = (
            (
                (u.reshape(n1, 1, nl))
                * (self._np.log(pi) - self._np.log(1 - pi)).reshape(1, nq, nl)
            ).sum(2)
            + self._np.log(alpha_1.reshape(1, nq))
            + (self._np.log(1 - pi) @ tau_2.T).sum(1)
        )

        # For computationnal stability reasons 1.
        l_tau_1 -= l_tau_1.max(axis=1).reshape(n1, 1)
        tau_1 = self._np.exp(l_tau_1)
        tau_1 /= tau_1.sum(axis=1).reshape(n1, 1)  # Normalize.

        # For computationnal stability reasons 2.
        tau_1[tau_1 < eps_1] = eps_1
        tau_1 /= tau_1.sum(axis=1).reshape(n1, 1)  # Re-Normalize.

        # Update of tau_2 with sparsity trick.
        l_tau_2 = (
            (
                (v.reshape(n2, nq, 1))
                * (self._np.log(pi) - self._np.log(1 - pi)).reshape(1, nq, nl)
            ).sum(1)
            + self._np.log(alpha_2.reshape(1, nl))
            + (tau_1 @ self._np.log(1 - pi)).sum(0)
        )

        # For computationnal stability reasons 1.
        l_tau_2 -= l_tau_2.max(axis=1).reshape(n2, 1)
        tau_2 = self._np.exp(l_tau_2)
        tau_2 /= tau_2.sum(axis=1).reshape(n2, 1)  # Normalize.

        # For computationnal stability reasons 2.
        tau_2[tau_2 < eps_2] = eps_2
        tau_2 /= tau_2.sum(axis=1).reshape(n2, 1)  # Re-Normalize.
        ########################## M-step  ##########################
        alpha_1 = tau_1.mean(0)
        alpha_2 = tau_2.mean(0)
        pi = (
            tau_1[indices_ones[0]].reshape(-1, nq, 1)
            * tau_2[indices_ones[1]].reshape(-1, 1, nl)
        ).sum(0) / (tau_1.sum(0).reshape(nq, 1) * tau_2.sum(0).reshape(1, nl))
        return pi, alpha_1, alpha_2, tau_1, tau_2

    def _compute_likelihood(
        self, indices_ones, pi, alpha_1, alpha_2, tau_1, tau_2
    ):
        """Compute the log-likelihood of the model with the given parameters.

        Parameters
        ----------
        indices_ones : Non zero indices of the data matrix.
        pi : Connection probability matrix between row and column groups.
        alpha_1 : Row group model parameters.
        alpha_2 : Column group model parameters.
        tau_1 : Row group variationnal parameters.
        tau_2 : Column group variationnal parameters.
        """
        nq, nl = self.n_row_clusters, self.n_column_clusters
        return (
            -self._np.sum(tau_1 * self._np.log(tau_1))
            - self._np.sum(tau_2 * self._np.log(tau_2))
            + tau_1.sum(0) @ self._np.log(alpha_1)
            + tau_2.sum(0) @ self._np.log(alpha_2).T
            + (
                tau_1[indices_ones[0]].reshape(-1, nq, 1)
                * tau_2[indices_ones[1]].reshape(-1, 1, nl)
                * (
                    self._np.log(pi.reshape(1, nq, nl))
                    - self._np.log(1 - pi).reshape(1, nq, nl)
                )
            ).sum()
            + (tau_1.sum(0) @ self._np.log(1 - pi) @ tau_2.sum(0))
        )

    def _init_LBM_random(self, n1, n2, nq, nl, nb_ones):
        """Randomly initialize the LBM model and variationnal parameters.

        Parameters
        ----------
        n1 : number of rows of the data matrix.
        n2 : number of column of the data matrix.
        nq : number of row clusters.
        nl : number of column clusters.
        """
        eps_1 = 1e-2 / n1
        eps_2 = 1e-2 / n2
        alpha_1 = (self._np.ones(nq) / nq).reshape((nq, 1))
        alpha_2 = (self._np.ones(nl) / nl).reshape((1, nl))
        tau_1 = self._np.random.uniform(size=(n1, nq)) ** 2
        tau_1 /= tau_1.sum(axis=1).reshape(n1, 1)
        tau_1[tau_1 < eps_1] = eps_1
        tau_1 /= tau_1.sum(axis=1).reshape(n1, 1)  # Re-Normalize.
        tau_2 = self._np.random.uniform(size=(n2, nl)) ** 2
        tau_2 /= tau_2.sum(axis=1).reshape(n2, 1)
        tau_2[tau_2 < eps_2] = eps_2
        tau_2 /= tau_2.sum(axis=1).reshape(n2, 1)  # Re-Normalize.
        pi = self._np.random.uniform(0, 1e-7, (nq, nl))
        pi = self._np.random.uniform(
            0.2 * nb_ones / (n1 * n2), 2 * nb_ones / (n1 * n2), (nq, nl)
        )
        return (alpha_1.flatten(), alpha_2.flatten(), tau_1, tau_2, pi)

    def __repr__(self):
        return f"""LBM(
                    n_row_clusters={self.n_row_clusters},
                    n_column_clusters={self.n_column_clusters},
                    max_iter={self.max_iter},
                    n_init={self.n_init},
                    n_init_total_run={self.n_init_total_run},
                    n_iter_early_stop={self.nb_iter_early_stop},
                    rtol={self.rtol},
                    atol={self.atol},
                    verbosity={self.verbosity},
                    use_gpu={self.use_gpu},
                    gpu_index={self.gpu_index},
                )"""

    def copy(self):
        """Returns a copy of the model.
        """
        model = LBM(
            n_row_clusters=self.n_row_clusters,
            n_column_clusters=self.n_column_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            n_init_total_run=self.n_init_total_run,
            n_iter_early_stop=self.nb_iter_early_stop,
            rtol=self.rtol,
            atol=self.atol,
            verbosity=self.verbosity,
            use_gpu=self.use_gpu,
            gpu_index=self.gpu_index,
        )
        model._nb_rows = self._nb_rows
        model._nb_cols = self._nb_cols
        model.loglikelihood_ = self.loglikelihood_
        model._np = self._np
        model._cupyx = self._cupyx
        model.trained_successfully_ = self.trained_successfully_
        model.pi_ = copy.copy(self.pi_)
        model.alpha_1_ = copy.copy(self.alpha_1_)
        model.alpha_2_ = copy.copy(self.alpha_2_)
        model.tau_1_ = copy.copy(self.tau_1_)
        model.tau_2_ = copy.copy(self.tau_2_)
        return model
