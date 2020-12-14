*****
SparseBM
*****
.. contents:: Table of Contents
    :local:

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Contents:

.. automodule:: sparsebm
   :members:

.. mdinclude:: ../../README.md

Class description
=================

Stochastic Block Model
----------------------
.. image:: sbm.png
        :scale: 50
        :align: center
        :alt: sbm

The SBM is encapsulated in the *SBM* class that inherits from the
**sklearn.base.BaseEstimator** that is the base class for all estimators
in scikit-learn.
A number of clusters should be specify with the parameter *n_clusters* otherwise
the default value 5 is used.
If the **cupy** module is installed, the class is using the GPU with higher
memory available. The parameter *use_gpu* can disable this behaviour and the parameter *gpu_index* can enforce the use of a specific GPU.

The class implements the random initializations strategy that corresponds to the run of *n_iter_early_stop* EM steps on *n_init* random initializations,
followed by iterations until convergence of the criterion for the *n_init_total_run* best results after these preliminary steps;
*n_iter_early_stop*, *n_init* and *n_init_total_run* being parameters of the class.

The convergence of the criterion :math:`J(q_\gamma, \theta)` is declared when

.. math::
    J^{(t)}(q_\gamma, \theta) - J^{(t-5)}(q_\gamma, \theta) \leq ( atol + rtol \cdot \lvert J^{(t)}(q_\gamma, \theta)\rvert),
with *atol* and *rtol* being respectively the absolute tolerance and the relative tolerance.

.. autoclass:: SBM
   :members:

   .. automethod:: __init__


Latent Block Model
------------------
.. image:: lbm.png
        :scale: 50
        :align: center
        :alt: lbm

The *LBM* class encapsulates the Latent Block Model and its random initialisation procedure.
Its usage is similar to the *SBM* class and we refer the reader to the previous section for more details.

.. autoclass:: LBM
   :members:

   .. automethod:: __init__

Model selection
---------------

The *ModelSelection* class encapsulates the model selection algorithm based on
the split and merge strategy. The argument *model_type* specifies the model to use
and *n_clusters_max* specifies an upper limit of the number of groups the algorithm can explore.
The split strategy stops when the number of classes is
greater than :math:`min(1.5 \cdot nnq\_best,\; nnq\_best + 10,\; n\_clusters\_max)`
with :math:`nnq\_best` being the number of classes of the best model found so far
during the split strategy.
The merge strategy stops when the minimum relevant number of classes is reached.
The split and merge strategy alternates until no best model is found for two iterations.

The argument *plot* specifies if an illustration is displayed to the user during the learning process.

.. autoclass:: ModelSelection
   :members:

   .. automethod:: __init__

Helper functions
================

Graph generation with LBM
-------------------------

The function *generate_SBM_dataset* generates a graph using the SBM with a specified number of nodes, a number of clusters, the cluster proportions, and the array of connection probabilities between classes.
The argument *symmetric* indicates if the adjacency matrix has to be symmetric.
The generated sparse adjacency matrix and the generated indicator matrix of the latent clusters are returned in a dictionary at keys *data* and *cluster_indicator*.
The graph generation is implemented such as the adjacency matrix X is created block after block and never manipulates dense matrices.


.. autofunction:: generate_LBM_dataset

Graph generation with SBM
-------------------------

The function *generate_SBM_dataset* generates a graph using the LBM.

.. autofunction:: generate_SBM_dataset

Co-classication adjusted rand index
----------------

.. autofunction:: CARI

.. mdinclude:: ../../experiments/README.md

* :ref:`genindex`
