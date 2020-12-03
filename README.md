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

## Example
### Generate SBM Synthetic graph
- Generate a synthetic graph to analyse with SBM:

```python
import numpy as np
from sparsebm import generate_SBM_dataset

# Specifying the parameters of the dataset to generate.
number_of_nodes = 10 ** 3
number_of_clusters = 4
cluster_proportions = (
    np.ones(number_of_clusters) / number_of_clusters
)  # Here equals classe sizes
connection_probabilities = np.array(
    [
        [0.05, 0.018, 0.006, 0.0307],
        [0.018, 0.037, 0, 0],
        [0.006, 0, 0.055, 0.012],
        [0.0307, 0, 0.012, 0.043],
    ]
)  # The probability of link between the classes. Here symmetric.

# Generate The dataset.
dataset = generate_SBM_dataset(
    number_of_nodes,
    number_of_clusters,
    connection_probabilities,
    cluster_proportions,
    symmetric=True,
)
graph = dataset["data"]
cluster_indicator = dataset["cluster_indicator"]
```

### Infere with sparsebm SBM:
 - Use the bernoulli Stochastic Bloc Model:
```python
    from sparsebm import SBM

    # instantiate the Stochastic Block Model class.
    model = SBM(
        number_of_clusters,  # A number of classes must be specify. Otherwise see model selection.
        n_init=50,  # Specifying the number of initializations to perform.
        n_iter_early_stop=30,  # Specifying the number of EM-steps to perform on each init.
        n_init_total_run=10,  # Specifying the number inits to keep and to train until convergence.
        verbosity=1,  # Either 0, 1 or 2. Higher value display more information to the user.
    )
    model.fit(graph, symmetric=True)
    print("Labels:", model.labels)
```
To use GPU acceleration, CUPY needs to be installed and replace gpu_number to the desired GPU index.
