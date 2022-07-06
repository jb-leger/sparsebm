# Getting started with SparseBM

SparseBM is a python module for handling sparse graphs with Block Models.
The module is an implementation of the variational inference algorithm for the Stochastic Block Model (SBM) and the Latent Block Model (LBM) for sparse graphs, which leverages the sparsity of edges to scale to very large numbers of nodes. The module can use [Cupy](https://cupy.dev/) to take advantage of the hardware acceleration provided by graphics processing units (GPU).

## Installing

The SparseBM module is distributed through the [PyPI repository](https://pypi.org/project/sparsebm/) and the documentation is available at [sparsebm.readthedocs.io](https://sparsebm.readthedocs.io/).


### With GPU acceleration (recommended if GPUs are available)

This option is recommended if GPUs are available to speedup computation.

With the package installer pip:

```
pip3 install sparsebm[gpu]
```

The [Cupy] module will be installed as a dependency.

[Cupy]: https://github.com/gfrisch/sparsebm

Alternatively [Cupy] can be installed separately, and will be used by `sparsebm`
if available.

```
pip3 install sparsebm
pip3 install cupy
```

### Without GPU acceleration

Without GPU acceleration, only CPUs are used. The infererence process still uses
sparsity, but no GPU linear algebra operations.

```
pip3 install sparsebm
```

For users who do not have GPU, we recommend the free serverless Jupyter notebook environment provided by [Google Colab](https://colab.research.google.com/) where the Cupy module is already installed and ready to be used with a GPU.

## Example with the Stochastic Block Model

- Generate a synthetic graph for analysis with SBM:

    ```python
    from sparsebm import generate_SBM_dataset

    dataset = generate_SBM_dataset(symmetric=True)
    graph = dataset["data"]
    cluster_indicator = dataset["cluster_indicator"]
    ```


- Infer with the Bernoulli Stochastic Bloc Model:

```python
    from sparsebm import SBM

    number_of_clusters = cluster_indicator.shape[1]

    # A number of classes must be specify. Otherwise see model selection.
    model = SBM(number_of_clusters)
    model.fit(graph, symmetric=True)
    print("Labels:", model.labels)
```

- Compute performance:

```python
    from sparsebm.utils import ARI
    ari = ARI(cluster_indicator.argmax(1), model.labels)
    print("Adjusted Rand index is {:.2f}".format(ari))
```


## Example with the Latent Block Model

- Generate a synthetic graph for analysis with LBM:

```python
    from sparsebm import generate_LBM_dataset

    dataset = generate_LBM_dataset()
    graph = dataset["data"]
    row_cluster_indicator = dataset["row_cluster_indicator"]
    column_cluster_indicator = dataset["column_cluster_indicator"]
```

 - Use the Bernoulli Latent Bloc Model:

```python
    from sparsebm import LBM

    number_of_row_clusters = row_cluster_indicator.shape[1]
    number_of_columns_clusters = column_cluster_indicator.shape[1]

    # A number of classes must be specify. Otherwise see model selection.
    model = LBM(
        number_of_row_clusters,
        number_of_columns_clusters,
        n_init_total_run=1,
    )
    model.fit(graph)
    print("Row Labels:", model.row_labels)
    print("Column Labels:", model.column_labels)
```

- Compute performance:

```python
    from sparsebm.utils import CARI
    cari = CARI(
        row_cluster_indicator.argmax(1),
        column_cluster_indicator.argmax(1),
        model.row_labels,
        model.column_labels,
    )
    print("Co-Adjusted Rand index is {:.2f}".format(cari))
```
