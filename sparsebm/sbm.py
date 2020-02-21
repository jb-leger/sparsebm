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


class SBM_bernouilli:
    """SBM with bernouilli distribution.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form

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
        n_clusters,
        symetric=False,
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
        self._n_clusters = n_clusters
        self._symetric = symetric
        self._nb_rows = None
        self._loglikelihood = -np.inf
        self._trained_successfully = False
        self._pi = None
        self._alpha = None
        self._tau = None
        self._run_number = 0
        self._nb_runs_to_perform = n_init

        self.use_gpu = False
        self.gpu_number = False
        if gpu_number is not None and gpu_number >= 0:
            if not _CUPY_INSTALLED:
                self._np = np
                self.gpu_number = None
                print("GPU not used as cupy library seems not to be installed", file=sys.stderr)
            elif not cupy.cuda.is_available():
                self._np = np
                self.gpu_number = None
                print("GPU not used as CUDA is not available", file=sys.stderr)
            else:
                self._np = cupy
                self.gpu_number = None
                cupy.cuda.Device(gpu_number).use()
                self.use_gpu = True

    @property
    def group_membership_probability(self):
        """array_like: Returns the group membership probabilities"""
        assert self._trained_successfully == True, "Model not trained successfully"
        return self._alpha

    @property
    def labels(self):
        """array_like: Returns the labels"""
        assert self._trained_successfully == True, "Model not trained successfully"
        return self._tau.argmax(1)

    @property
    def predict_proba(self):
        """array_like: Returns the predicted classes membership probabilities"""
        assert self._trained_successfully == True, "Model not trained successfully"
        return self._tau

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
        assert self._trained_successfully == True, "Model must be trained successfully before"
        return (
            self._loglikelihood
            - (self._n_clusters - 1) / 2 * np.log(self._nb_rows)
            - (self._n_clusters**2)/ 2 * np.log(self._nb_rows*(self._nb_rows-1))
        )

    def fit(self, X):
        """Perform co-clustering by direct maximization of graph modularity.

        Parameters
        ----------
        X : scipy sparse matrix, shape=(n_nodes, n_nodes)
            Matrix to be analyzed
        use_gpu : False or int
            If not false, use the gpu index specified
        """
        self._trained_successfully = False
        n, n2 = X.shape
        assert n == n2, "Entry matrix is not squared"
        self._nb_rows = n
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
                (success, ll, pi, alpha, tau) = self._fit_single(
                    indices_ones, n, early_stop=self.nb_iter_early_stop
                )
                calculation_result = [
                    ll,
                    next(tiebreaker),
                    pi,
                    alpha,
                    tau,
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

                (pi, alpha, tau) = (
                    init[2],
                    init[3],
                    init[4],
                )

                (success, ll, pi, alpha, tau) = self._fit_single(
                    indices_ones,
                    n,
                    init_params=(pi, alpha, tau,),
                )

                if success and ll > self._loglikelihood:
                    self._loglikelihood = ll.get() if self.use_gpu else ll
                    self._trained_successfully = True
                    self._pi = pi.get() if self.use_gpu else pi
                    self._alpha = alpha.get() if self.use_gpu else alpha
                    self._tau = tau.get() if self.use_gpu else tau
        except KeyboardInterrupt:
            pass
        finally:
            if self.verbosity > 0:
                bar.finish()
        if self._trained_successfully:
            print("Trained.")

    def _fit_single(self, indices_ones, n, early_stop=None, init_params=None):
        """Perform one run of the SBM_bernouilli algorithm with one random initialization.

        Parameters
        ----------
        indices_ones : Non zero indices of the data matrix.
        n1 : Number of rows in the data matrix.
        """
        old_ll = -self._np.inf
        success = False

        if init_params:
            (pi, alpha, tau) = init_params
        else:
            alpha, tau, pi = self._init_bernouilli_SBM_random(
                n, self._n_clusters, len(indices_ones)
            )

        # Repeat EM step until convergence.
        for iteration in range(self.max_iter):
            if early_stop and iteration >= early_stop:
                ll = self._compute_likelihood(indices_ones, pi, alpha, tau)
                break
            if iteration % 10 == 0:
                ll = self._compute_likelihood(indices_ones, pi, alpha, tau)
                if self._np.abs(old_ll - ll) < self.tol:
                    success = True
                    break
                if self.verbosity > 2:

                    print(
                        f"\t EM Iter: {iteration:5d}  \t  log-like:{ll.get() if self.use_gpu else ll:.4f} \t diff:{self._np.abs(old_ll - ll).get() if self.use_gpu else self._np.abs(old_ll - ll):.6f}"
                    )
                old_ll = ll
            pi, alpha, tau= self._step_EM(indices_ones, pi, alpha, tau, n)
        else:
            success = True
        if self.verbosity > 1:
            print(
                f"Run {self._run_number:3d} / {self._nb_runs_to_perform:3d} \t success : {success} \t log-like: {ll.get()  if self.use_gpu else ll:.4f} \t nb_iter: {iteration:5d}"
            )

        return success, ll, pi, alpha, tau

    def _step_EM(self, indices_ones, pi, alpha, tau, n1):
        """Realize EM step. Update both variationnal and model parameters.

        Parameters
        ----------
        indices_ones : Non zero indices of the data matrix.
        pi : Connection probability matrix between row and column groups.
        alpha : Group model parameters.
        tau : Group variationnal parameters.
        n : Number of rows in the data matrix.
        """
        eps = 1e-2 / n1
        nq = self._n_clusters

        ########################## E-step  ##########################

        # Precomputations needed.
        u = self._np.zeros((n1, nq))
        if self.use_gpu:
            self._np.scatter_add(u, indices_ones[0], tau[indices_ones[1]])
        else:
            self._np.add.at(u, indices_ones[0], tau[indices_ones[1]])

        # Update of tau_1 with sparsity trick.
        l_tau = (
            (
                (u.reshape(n1, 1, nq))
                * (self._np.log(pi) - self._np.log(1 - pi)).reshape(1, nq, nq)
            ).sum(2)
            + self._np.log(alpha.reshape(1, nq))
            + tau.sum(0) @ np.log(1-pi.T)
            - tau @ np.log(1-pi.T)
        )

        # For computationnal stability reasons 1.
        l_tau -= l_tau.max(axis=1).reshape(n1, 1)
        tau = self._np.exp(l_tau)
        tau /= tau.sum(axis=1).reshape(n1, 1)  # Normalize.

        # For computationnal stability reasons 2.
        tau[tau < eps] = eps
        tau /= tau.sum(axis=1).reshape(n1, 1)  # Re-Normalize.

        ########################## M-step  ##########################
        alpha = tau.mean(0)
        tau_sum = tau.sum(0)
        pi = (
            tau[indices_ones[0]].reshape(-1, nq, 1)
            * tau[indices_ones[1]].reshape(-1, 1, nq)
        ).sum(0) / ((tau_sum.reshape((-1,1))* tau_sum) - tau.T @ tau )

        return pi, alpha, tau,

    def _compute_likelihood(self, indices_ones, pi, alpha, tau):
        """Compute the log-likelihood of the model with the given parameters.

        Parameters
        ----------
        indices_ones : Non zero indices of the data matrix.
        pi : Connection probability matrix between row and column groups.
        alpha : Group model parameters.
        tau : Group variationnal parameters.
        """
        nq = self._n_clusters
        tau_sum = tau.sum(0)
        ll = (
            -self._np.sum(tau * self._np.log(tau))
            + tau.sum(0) @ self._np.log(alpha)
            + (
                tau[indices_ones[0]].reshape(-1, nq, 1)
                * tau[indices_ones[1]].reshape(-1, 1, nq)
                * (
                    self._np.log(pi.reshape(1, nq, nq))
                    - self._np.log(1 - pi).reshape(1, nq, nq)
                )
            ).sum()
            + (((tau_sum.reshape((-1,1))* tau_sum) - tau.T @ tau ) * self._np.log(1-pi)).sum()
        )
        return ll/2 if self._symetric else ll

    def _init_bernouilli_SBM_random(self, n1, nq, nb_ones):
        """Randomly initialize the SBM bernouilli model and variationnal parameters.

        Parameters
        ----------
        n1 : number of rows of the data matrix.
        nq : number of clusters.
        """
        eps = 1e-2 / n1

        alpha = (self._np.ones(nq) / nq).reshape((nq, 1))

        tau = self._np.random.uniform(size=(n1, nq)) ** 2
        tau /= tau.sum(axis=1).reshape(n1, 1)
        tau[tau < eps] = eps
        tau /= tau.sum(axis=1).reshape(n1, 1)  # Re-Normalize.
        pi = self._np.random.uniform(0, 2 * nb_ones / (n1 * n1), (nq, nq))
        if self._symetric:
            pi = (pi@pi.T)/2

        return (alpha.flatten(), tau, pi)

    def __repr__(self):
        return f"""LBM_bernouilli(
                    n_clusters={self._n_clusters},
                    max_iter={self.max_iter},
                    n_init={self.n_init},
                    n_init_total_run={self.n_init_total_run},
                    n_iter_early_stop={self.nb_iter_early_stop},
                    tol={self.tol},
                    verbosity={self.verbosity},
                    gpu_number={self.gpu_number},
                )"""
