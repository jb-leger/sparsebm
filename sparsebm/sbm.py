import sys
import copy
import progressbar
import numpy as np
from heapq import heappush, heappushpop
from itertools import count
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator

try:
    import cupy
    import cupyx
    import GPUtil

    _CUPY_INSTALLED = True
    _DEFAULT_USE_GPU = True
except ImportError:
    _CUPY_INSTALLED = False
    _DEFAULT_USE_GPU = False


class SBM_bernouilli(BaseEstimator):
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

    use_gpu : bool, optional, default: _DEFAULT_USE_GPU
        Specify if a GPU should be used.

    gpu_index : int, optional, default: None
        Specify the gpu index if needed.

    Attributes
    ----------
    max_iter : int
        Maximum number of EM iterations
    n_init : int
        Number of initializations that will be run for n_iter_early_stop EM iterations.
    n_init_total_run : int
        Number of the n_init best initializations that will be run until convergence.
    n_iter_early_stop : int
        Number of EM iterations to used to run the n_init initializations.
    tol : int
        Tolerance to declare convergence
    verbosity : int
        Degree of verbosity. Scale from 0 (no message displayed) to 3.
    """

    def __init__(
        self,
        n_clusters=5,
        max_iter=10000,
        n_init=100,
        n_init_total_run=10,
        n_iter_early_stop=100,
        tol=1e-5,
        verbosity=1,
        use_gpu=_DEFAULT_USE_GPU,
        gpu_index=None,
    ):
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_init_total_run = (
            n_init_total_run if n_init > n_init_total_run else n_init
        )
        self.n_iter_early_stop = n_iter_early_stop
        self.tol = tol
        self.verbosity = verbosity
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index
        self.symetric = False

    def score(self, X, y=None, symetric=False):
        if not hasattr(self, "loglikelihood_"):
            self.fit(X, symetric=symetric)
        return self.get_ICL()

    def get_params(self, deep=True):
        return {
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "n_init_total_run": self.n_init_total_run,
            "n_iter_early_stop": self.n_iter_early_stop,
            "tol": self.tol,
            "verbosity": self.verbosity,
            "n_clusters": self.n_clusters,
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
            print(
                "GPU not used as cupy library seems not to be installed or CUDA is not available",
                file=sys.stderr,
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
                    print("GPU not used as no gpu is free", file=sys.stderr)
                else:
                    self._np = cupy
                    self._cupyx = cupyx
                    gpu_number = free_idx[0]
                    cupy.cuda.Device(gpu_number).use()

    @property
    def group_membership_probability(self):
        """array_like: Returns the group membership probabilities"""
        assert (
            self.trained_successfully_ == True
        ), "Model not trained successfully"
        return self.alpha_

    @property
    def labels(self):
        """array_like: Returns the labels"""
        assert (
            self.trained_successfully_ == True
        ), "Model not trained successfully"
        return self.tau_.argmax(1)

    @property
    def predict_proba(self):
        """array_like: Returns the predicted classes membership probabilities"""
        assert (
            self.trained_successfully_ == True
        ), "Model not trained successfully"
        return self.tau_

    @property
    def trained_successfully(self):
        """bool: Returns the predicted column classes membership probabilities"""
        return self.trained_successfully_

    def get_ICL(self):
        """Computation of the ICL criteria that can be used for model selection.
        Returns
        -------
        float
            value of the ICL criteria.
        """
        assert (
            self.trained_successfully_ == True
        ), "Model must be trained successfully before"
        return (
            self.loglikelihood_
            - (self.n_clusters - 1) / 2 * np.log(self._nb_rows)
            - (self.n_clusters ** 2)
            / 2
            * np.log(self._nb_rows * (self._nb_rows - 1))
        )

    def fit(self, X, y=None, symetric=False):
        """Perform co-clustering by direct maximization of graph modularity.

        Parameters
        ----------
        X : scipy sparse matrix, shape=(n_nodes, n_nodes)
            Matrix to be analyzed
        """
        self.symetric = symetric
        self._check_params()
        self.trained_successfully_ = False
        n, n2 = X.shape
        assert n == n2, "Entry matrix is not squared"
        self._nb_rows = n
        indices_ones = self._np.asarray(list(X.nonzero()))
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
                (success, ll, pi, alpha, tau) = self._fit_single(
                    indices_ones,
                    n,
                    early_stop=self.n_iter_early_stop,
                    run_number=run_number,
                )
                calculation_result = [ll, next(tiebreaker), pi, alpha, tau]
                if len(best_inits) < max(1, int(self.n_init_total_run)):
                    heappush(best_inits, calculation_result)
                else:
                    heappushpop(best_inits, calculation_result)
            if self.verbosity > 0:
                bar.finish()
                print(
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

                (pi, alpha, tau) = (init[2], init[3], init[4])

                (success, ll, pi, alpha, tau) = self._fit_single(
                    indices_ones,
                    n,
                    init_params=(pi, alpha, tau),
                    run_number=run_number,
                )

                if success and ll > self.loglikelihood_:
                    self.loglikelihood_ = ll.get() if self.use_gpu else ll
                    self.trained_successfully_ = True
                    self.pi_ = pi.get() if self.use_gpu else pi
                    self.alpha_ = alpha.get() if self.use_gpu else alpha
                    self.tau_ = tau.get() if self.use_gpu else tau
        except KeyboardInterrupt:
            pass
        finally:
            if self.verbosity > 0:
                bar.finish()
        return self

    def _fit_single(
        self,
        indices_ones,
        n,
        early_stop=None,
        init_params=None,
        in_place=False,
        run_number=None,
    ):
        """Perform one run of the SBM_bernouilli algorithm with one random initialization.

        Parameters
        ----------
        indices_ones : Non zero indices of the data matrix.
        n1 : Number of rows in the data matrix.
        """
        old_ll = -self._np.inf
        success = False

        if init_params:
            if init_params is True:
                if (
                    self.pi_ is not None
                    and self.alpha_ is not None
                    and self.tau_ is not None
                ):
                    alpha, tau, pi = (
                        self._np.asarray(self.alpha_),
                        self._np.asarray(self.tau_),
                        self._np.asarray(self.pi_),
                    )
                else:
                    assert False
            else:
                (pi, alpha, tau) = init_params
        else:
            alpha, tau, pi = self._init_bernouilli_SBM_random(
                n, self.n_clusters, len(indices_ones)
            )

        # Repeat EM step until convergence.
        for iteration in range(self.max_iter):
            if early_stop and iteration >= early_stop:
                ll = self._compute_likelihood(indices_ones, pi, alpha, tau)
                break
            if iteration % 5 == 0:
                ll = self._compute_likelihood(indices_ones, pi, alpha, tau)
                if self._np.abs(old_ll - ll) < self.tol:
                    success = True
                    break
                if self.verbosity > 2:

                    print(
                        f"\t EM Iter: {iteration:5d}  \t  log-like:{ll.get() if self.use_gpu else ll:.4f} \t diff:{self._np.abs(old_ll - ll).get() if self.use_gpu else self._np.abs(old_ll - ll):.6f}"
                    )
                old_ll = ll
            pi, alpha, tau = self._step_EM(indices_ones, pi, alpha, tau, n)
        else:
            success = True
        if self.verbosity > 1 and run_number:
            print(
                f"Run {run_number:3d} / {self.n_init:3d} \t success : {success} \t log-like: {ll.get()  if self.use_gpu else ll:.4f} \t nb_iter: {iteration:5d}"
            )

        if in_place:
            self.loglikelihood_ = ll.get() if self.use_gpu else ll
            self.trained_successfully_ = True
            self.pi_ = pi.get() if self.use_gpu else pi
            self.alpha_ = alpha.get() if self.use_gpu else alpha
            self.tau_ = tau.get() if self.use_gpu else tau

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
        nq = self.n_clusters

        ########################## E-step  ##########################

        # Precomputations needed.
        u = self._np.zeros((n1, nq))
        if self.use_gpu:
            self._cupyx.scatter_add(u, indices_ones[0], tau[indices_ones[1]])
        else:
            self._np.add.at(u, indices_ones[0], tau[indices_ones[1]])

        # Update of tau_1 with sparsity trick.
        l_tau = (
            (
                (u.reshape(n1, 1, nq))
                * (self._np.log(pi) - self._np.log(1 - pi)).reshape(1, nq, nq)
            ).sum(2)
            + self._np.log(alpha.reshape(1, nq))
            + tau.sum(0) @ np.log(1 - pi.T)
            - tau @ np.log(1 - pi.T)
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
        ).sum(0) / ((tau_sum.reshape((-1, 1)) * tau_sum) - tau.T @ tau)

        return pi, alpha, tau

    def _compute_likelihood(self, indices_ones, pi, alpha, tau):
        """Compute the log-likelihood of the model with the given parameters.

        Parameters
        ----------
        indices_ones : Non zero indices of the data matrix.
        pi : Connection probability matrix between row and column groups.
        alpha : Group model parameters.
        tau : Group variationnal parameters.
        """
        nq = self.n_clusters
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
            + (
                ((tau_sum.reshape((-1, 1)) * tau_sum) - tau.T @ tau)
                * self._np.log(1 - pi)
            ).sum()
        )
        return ll / 2 if self.symetric else ll

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
        pi = self._np.random.uniform(
            2 * nb_ones / (n1 * n1) / 10, 2 * nb_ones / (n1 * n1), (nq, nq)
        )
        if self.symetric:
            pi = (pi @ pi.T) / 2

        return (alpha.flatten(), tau, pi)

    def __repr__(self):
        return f"""SBM_bernouilli(
                    n_clusters={self.n_clusters},
                    max_iter={self.max_iter},
                    n_init={self.n_init},
                    n_init_total_run={self.n_init_total_run},
                    n_iter_early_stop={self.n_iter_early_stop},
                    tol={self.tol},
                    verbosity={self.verbosity},
                    use_gpu={self.use_gpu},
                    gpu_index={self.gpu_index},
                )"""

    def copy(self):
        """Returns a copy of the model.
        """
        model = SBM_bernouilli(
            self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            n_init_total_run=self.n_init_total_run,
            n_iter_early_stop=self.n_iter_early_stop,
            tol=self.tol,
            verbosity=self.verbosity,
            use_gpu=self.use_gpu,
        )
        model.symetric = self.symetric
        model._np = self._np
        model._cupyx = self._cupyx
        model._nb_rows = self._nb_rows
        model.loglikelihood_ = self.loglikelihood_
        model.trained_successfully_ = self.trained_successfully_
        model.pi_ = copy.copy(self.pi_)
        model.alpha_ = copy.copy(self.alpha_)
        model.tau_ = copy.copy(self.tau_)

        return model


if __name__ == "__main__":
    from sparsebm import generate_bernouilli_SBM_dataset
    import numpy as np

    number_of_nodes = 10 ** 3
    number_of_clusters = 4
    cluster_proportions = np.ones(number_of_clusters) / number_of_clusters
    connection_probabilities = (
        np.array(
            [
                [0.05, 0.018, 0.006, 0.0307],
                [0.018, 0.037, 0, 0],
                [0.006, 0, 0.055, 0.012],
                [0.0307, 0, 0.012, 0.043],
            ]
        )
        * 2
    )
    dataset = generate_bernouilli_SBM_dataset(
        number_of_nodes,
        number_of_clusters,
        connection_probabilities,
        cluster_proportions,
        symetric=True,
    )

    from sparsebm import SBM_bernouilli
    import sklearn
    from sklearn import metrics

    graph = dataset["data"]
    clusters_index = dataset["cluster_indicator"].argmax(1)

    model = SBM_bernouilli(verbosity=0)
    train = test = np.arange(number_of_nodes)
    n_clusters = np.arange(1, 10)
    clf = sklearn.model_selection.GridSearchCV(
        estimator=model,
        n_jobs=4,
        param_grid={"n_clusters": list(range(2, 5))},
        cv=[[train, test]],
    )
    print("Start grid search algorithm")
    clf.fit(graph, symetric=True)
    print(
        "Best number of classes is {} according to ICL".format(
            clf.best_params_
        )
    )
    ari = metrics.adjusted_rand_score(
        clusters_index, clf.best_estimator_.labels
    )
    print("Adjusted Rand Index is {}".format(ari))
