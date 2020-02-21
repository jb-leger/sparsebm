import sys
import progressbar
import numpy as np
from heapq import heappush, heappushpop
from itertools import count

try:
    import cupy

    _CUPY_INSTALLED = True
except ImportError:
    _CUPY_INSTALLED = False


class LBM_bernouilli:
    """LBM with bernouilli distribution.

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

    tol : float, default: 1e-5
        Tolerance to declare convergence

    verbosity : int, optional, default: 1
        Degree of verbosity. Scale from 0 (no message displayed) to 3.

    gpu_number : int, optional, default: 1
        Select the index of the GPU. None if no need of GPU.

    Attributes
    ----------
    max_iter : int
        Maximum number of EM iterations
    n_init : int
        Number of initializations that will be run for n_iter_early_stop EM iterations.
    n_init_total_run : int
        Number of the n_init best initializations that will be run until convergence.
    nb_iter_early_stop : int
        Number of EM iterations to used to run the n_init initializations.
    tol : int
        Tolerance to declare convergence
    verbosity : int
        Degree of verbosity. Scale from 0 (no message displayed) to 3.
    """

    def __init__(
        self,
        n_row_clusters,
        n_column_clusters,
        max_iter=10000,
        n_init=100,
        n_init_total_run=10,
        n_iter_early_stop=100,
        tol=1e-5,
        verbosity=1,
        gpu_number=None,
    ):
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_init_total_run = (
            n_init_total_run if n_init > n_init_total_run else n_init
        )
        self.nb_iter_early_stop = n_iter_early_stop
        self.tol = tol
        self.verbosity = verbosity

        self._np = np
        self._n_row_clusters = n_row_clusters
        self._n_column_clusters = n_column_clusters
        self._nb_rows = None
        self._nb_cols = None
        self._loglikelihood = -np.inf
        self._trained_successfully = False
        self._pi = None
        self._alpha_1 = None
        self._alpha_2 = None
        self._tau_1 = None
        self._tau_2 = None
        self._run_number = 0
        self._nb_runs_to_perform = n_init

        self.gpu_number = None
        self.use_gpu = False
        if gpu_number is not None and gpu_number >= 0:
            if not _CUPY_INSTALLED:
                self._np = np
                self.gpu_number = None
                print("GPU not used as cupy library seems not to be installed")
            elif not cupy.cuda.is_available():
                self._np = np
                self.gpu_number = None
                print("GPU not used as CUDA is not available")
            else:
                self._np = cupy
                cupy.cuda.Device(gpu_number).use()
                self.use_gpu = True
                self.gpu_number = gpu_number

    @property
    def row_group_membership_probability(self):
        """array_like: Returns the row group membership probabilities"""
        assert self._trained_successfully == True, "Model not trained successfully"
        return self._alpha_1

    @property
    def column_group_membership_probability(self):
        """array_like: Returns the column group membership probabilities"""
        assert self._trained_successfully == True, "Model not trained successfully"
        return self._alpha_2

    @property
    def row_labels(self):
        """array_like: Returns the row labels"""
        assert self._trained_successfully == True, "Model not trained successfully"
        return self._tau_1.argmax(1)

    @property
    def column_labels(self):
        """array_like: Returns the column labels"""
        assert self._trained_successfully == True, "Model not trained successfully"
        return self._tau_2.argmax(1)

    @property
    def row_predict_proba(self):
        """array_like: Returns the predicted row classes membership probabilities"""
        assert self._trained_successfully == True
        return self._tau_1

    @property
    def column_predict_proba(self):
        """array_like: Returns the predicted column classes membership probabilities"""
        assert self._trained_successfully == True, "Model not trained successfully"
        return self._tau_2

    @property
    def trained_successfully(self):
        """bool: Returns the predicted column classes membership probabilities"""
        return self._trained_successfully

    def get_ICL(self):
        """Computation of the ICL criteria that can be used for model selection.
        Returns
        -------
        float
            value of the ICL criteria.
        """
        assert self._trained_successfully == True, "Model not trained successfully"
        return (
            self._loglikelihood
            - (self._n_row_clusters - 1) / 2 * np.log(self._nb_rows)
            - (self._n_column_clusters - 1) / 2 * np.log(self._nb_cols)
            - (self._n_column_clusters * self._n_row_clusters)
            / 2
            * np.log(self._nb_cols * self._nb_rows)
        )

    def fit(self, X):
        """Perform co-clustering by direct maximization of graph modularity.

        Parameters
        ----------
        X : numpy matrix or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        n1, n2 = X.shape
        self._nb_rows = n1
        self._nb_cols = n2
        indices_ones = self._np.asarray(X.nonzero())
        try:
            # Initialize and start to run each for a while.

            if self.verbosity > 0:
                print("---------- START RANDOM INITIALIZATIONS ---------- ")
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
                self._run_number = run_number
                (success, ll, pi, alpha_1, alpha_2, tau_1, tau_2) = self._fit_single(
                    indices_ones, n1, n2, early_stop=self.nb_iter_early_stop
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
                print("---------- START TRAINING BEST INITIALIZATIONS ---------- ")
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
            self._nb_runs_to_perform = len(best_inits)
            for run_number, init in enumerate(best_inits):
                self._run_number = run_number
                if self.verbosity > 0:
                    bar.update(run_number)

                (pi, alpha_1, alpha_2, tau_1, tau_2) = (
                    init[2],
                    init[3],
                    init[4],
                    init[5],
                    init[6],
                )
                (success, ll, pi, alpha_1, alpha_2, tau_1, tau_2) = self._fit_single(
                    indices_ones,
                    n1,
                    n2,
                    init_params=(pi, alpha_1, alpha_2, tau_1, tau_2),
                )

                if success and ll > self._loglikelihood:
                    self._loglikelihood = ll.get() if self.use_gpu else ll
                    self._trained_successfully = True
                    self._pi = pi.get() if self.use_gpu else pi
                    self._alpha_1 = alpha_1.get() if self.use_gpu else alpha_1
                    self._alpha_2 = alpha_2.get() if self.use_gpu else alpha_2
                    self._tau_1 = tau_1.get() if self.use_gpu else tau_1
                    self._tau_2 = tau_2.get() if self.use_gpu else tau_2
        except KeyboardInterrupt:
            pass
        finally:
            if self.verbosity > 0:
                bar.finish()
        if self._trained_successfully:
            print("Trained.")

    def _fit_single(self, indices_ones, n1, n2, early_stop=None, init_params=None):
        """Perform one run of the LBM_bernouilli algorithm with one random initialization.

        Parameters
        ----------
        indices_ones : Non zero indices of the data matrix.
        n1 : Number of rows in the data matrix.
        n2 : Number of columns in the data matrix.
        """
        old_ll = -self._np.inf
        success = False

        if init_params:
            (pi, alpha_1, alpha_2, tau_1, tau_2) = init_params
        else:
            alpha_1, alpha_2, tau_1, tau_2, pi = self._init_bernouilli_LBM_random(
                n1, n2, self._n_row_clusters, self._n_column_clusters, len(indices_ones)
            )

        # Repeat EM step until convergence.
        for iteration in range(self.max_iter):
            if early_stop and iteration >= early_stop:
                ll = self._compute_likelihood(
                    indices_ones, pi, alpha_1, alpha_2, tau_1, tau_2
                )
                break
            if iteration % 10 == 0:
                ll = self._compute_likelihood(
                    indices_ones, pi, alpha_1, alpha_2, tau_1, tau_2
                )
                if self._np.abs(old_ll - ll) < self.tol:
                    success = True
                    break
                if self.verbosity > 2:

                    print(
                        f"\t EM Iter: {iteration:5d}  \t  log-like:{ll.get() if self.use_gpu else ll:.4f} \t diff:{self._np.abs(old_ll - ll).get() if self.use_gpu else self._np.abs(old_ll - ll):.6f}"
                    )
                old_ll = ll
            pi, alpha_1, alpha_2, tau_1, tau_2 = self._step_EM(
                indices_ones, pi, alpha_1, alpha_2, tau_1, tau_2, n1, n2
            )
        else:
            success = True
        if self.verbosity > 1:
            print(
                f"Run {self._run_number:3d} / {self._nb_runs_to_perform:3d} \t success : {success} \t log-like: {ll.get()  if self.use_gpu else ll:.4f} \t nb_iter: {iteration:5d}"
            )

        return success, ll, pi, alpha_1, alpha_2, tau_1, tau_2

    def _step_EM(self, indices_ones, pi, alpha_1, alpha_2, tau_1, tau_2, n1, n2):
        """Realize EM step. Update both variationnal and model parameters.

        Parameters
        ----------
        indices_ones : Non zero indices of the data matrix.
        pi : Connection probability matrix between row and column groups.
        alpha_1 : Row group model parameters.
        alpha_2 : Column group model parameters.
        tau_1 : Row group variationnal parameters.
        tau_2 : Column group variationnal parameters.
        n1 : Number of rows in the data matrix.
        n2 : Number of columns in the data matrix.
        """
        eps_1 = 1e-2 / n1
        eps_2 = 1e-2 / n2
        nq, nl = self._n_row_clusters, self._n_column_clusters

        ########################## E-step  ##########################

        # Precomputations needed.
        u = self._np.zeros((n1, nl))
        v = self._np.zeros((n2, nq))
        if self.use_gpu:
            self._np.scatter_add(u, indices_ones[0], tau_2[indices_ones[1]])
            self._np.scatter_add(v, indices_ones[1], tau_1[indices_ones[0]])
        else:
            self._np.add.at(u, indices_ones[0], tau_2[indices_ones[1]])
            self._np.add.at(v, indices_ones[1], tau_1[indices_ones[0]])

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

    def _compute_likelihood(self, indices_ones, pi, alpha_1, alpha_2, tau_1, tau_2):
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
        nq, nl = self._n_row_clusters, self._n_column_clusters
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

    def _init_bernouilli_LBM_random(self, n1, n2, nq, nl, nb_ones):
        """Randomly initialize the LBM bernouilli model and variationnal parameters.

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
        pi = self._np.random.uniform(0, 2 * nb_ones / (n1 * n2), (nq, nl))
        return (alpha_1.flatten(), alpha_2.flatten(), tau_1, tau_2, pi)

    def __repr__(self):
        return f"""LBM_bernouilli(
                    n_row_clusters={self._n_row_clusters},
                    n_column_clusters={self._n_column_clusters},
                    max_iter={self.max_iter},
                    n_init={self.n_init},
                    n_init_total_run={self.n_init_total_run},
                    n_iter_early_stop={self.nb_iter_early_stop},
                    tol={self.tol},
                    verbosity={self.verbosity},
                    gpu_number={self.gpu_number},
                )"""
