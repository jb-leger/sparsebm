# Getting started with SparseBM

SparseBM is a python module for handling sparse graphs with Block Models.
The module is an implementation of the variational inference algorithm for the stochastic block model and the latent block model for sparse graphs, which leverages on the sparsity of edges to scale upto a very large number of nodes. The module can use [Cupy](https://cupy.dev/) to take advantage of the hardware speed up provided by graphics processingunits (GPU).

## Installing

SparseBMmodule is distributed through the [PyPI repository](https://pypi.org/project/sparsebm/) and the documentation is available at [sparsebm.readthedocs.io](https://sparsebm.readthedocs.io/). The module can be installed with the package installer pip:

```
pip3 install sparsebm
```

:warning: **To leverage GPU** accelaration, the  [Cupy](https://cupy.dev/) module **must** be installed with pip or anaconda or directly with the extra argument when installing SparseBM:

```
pip3 install sparsebm[gpu]
```

Or directly with
```
pip3 install sparsebm
pip3 install cupy
```

For users that do not have GPU, we advise the free serverless Jupyter notebook environment provided by [Google Colab](https://colab.research.google.com/) where the Cupy module is already installed and ready to use with one GPU.

## Example with Stochastic Block Model

- Generate a synthetic graph to analyse with SBM:

```python
from sparsebm import generate_SBM_dataset

dataset = generate_SBM_dataset(symmetric=True)
graph = dataset["data"]
cluster_indicator = dataset["cluster_indicator"]
```


- Infere with the bernoulli Stochastic Bloc Model:

```python
    from sparsebm import SBM

    number_of_clusters = cluster_indicator.shape[1]

    # A number of classes must be specify. Otherwise see model selection.
    model = SBM(number_of_clusters)
    model.fit(graph, symmetric=True)
    print("Labels:", model.labels)
```

- Compute performances:

```python
    from sparsebm.utils import ARI
    ari = ARI(cluster_indicator.argmax(1), model.labels)
    print("Adjusted Rand index is {:.2f}".format(ari))
```


## Example with Latent Block Model

- Generate a synthetic graph to analyse with LBM:

```python
from sparsebm import generate_LBM_dataset

dataset = generate_LBM_dataset()
graph = dataset["data"]
row_cluster_indicator = dataset["row_cluster_indicator"]
column_cluster_indicator = dataset["column_cluster_indicator"]
```

 - Use the bernoulli Latent Bloc Model:

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

- Compute performances:

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
