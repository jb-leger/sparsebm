# SparseBM: a python module for handling sparse graphs with Block Models

## Installing

From pypi:

```
pip3 install sparsebm
```

To use GPU acceleration:

```
pip3 install sparsebm[gpu]
```

Or
```
pip3 install sparsebm
pip3 install cupy
```

## Example with Stochastic Block Model
### Generate SBM Synthetic graph
- Generate a synthetic graph to analyse with SBM:

```python
from sparsebm import generate_SBM_dataset

dataset = generate_SBM_dataset(symmetric=True)
graph = dataset["data"]
cluster_indicator = dataset["cluster_indicator"]
```

### Infere with sparsebm SBM:
 - Use the bernoulli Stochastic Bloc Model:
```python
    from sparsebm import SBM

    number_of_clusters = cluster_indicator.shape[1]

    # A number of classes must be specify. Otherwise see model selection.
    model = SBM(number_of_clusters)
    model.fit(graph, symmetric=True)
    print("Labels:", model.labels)
```

### Compute performances:
```python
    from sparsebm.utils import ARI
    ari = ARI(cluster_indicator.argmax(1), model.labels)
    print("Adjusted Rand index is {:.2f}".format(ari))
```

To use GPU acceleration, CUPY needs to be installed and replace gpu_number to the desired GPU index.

## Example with Latent Block Model

### Generate LBM Synthetic graph
- Generate a synthetic graph to analyse with LBM:

```python
from sparsebm import generate_LBM_dataset

dataset = generate_LBM_dataset()
graph = dataset["data"]
row_cluster_indicator = dataset["row_cluster_indicator"]
column_cluster_indicator = dataset["column_cluster_indicator"]
```

### Infere with sparsebm LBM:
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

### Compute performances:
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
