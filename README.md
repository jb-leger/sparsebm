# sparsebm: a python implementation for the Latent Bloc Model (LBM) and Stochastic Bloc Model (SBM) for an efficient analysis of large graphs.

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
from sparsebm import generate_bernouilli_SBM
#
# Define the properties of your graph
#
number_of_nodes = 10 ** 3
number_of_clusters = 4
cluster_proportions = np.ones(number_of_clusters) / number_of_clusters
connection_probabilities = np.array([[0.05, 0.018, 0.006, 0.0307], [0.018, 0.037, 0, 0], [0.006, 0, 0.055, 0.012], [0.0307, 0, 0.012, 0.043]])
#
# The graph is generated
#
data = generate_bernouilli_SBM(number_of_nodes, number_of_clusters, connection_probabilities, cluster_proportions, symetric=True)
graph, cluster_indicator, = (data["X"], data["Y"])
```

### Infere with sparsebm SBM_bernouilli:
 - Use the bernouilli Stochastic Bloc Model:
```python
    from sparsebm import SBM_bernouilli

    model = SBM_bernouilli(
        number_of_clusters,
        gpu_number=None, # Or give the desired GPU index.
        symetric=True,
    )

    model.fit(graph)

    print("Labels:")
    print(model.labels)
```
To use GPU acceleration, CUPY needs to be installed and replace gpu_number to the desired GPU index.



### Generate LBM Synthetic graph
- Generate a synthetic graph to analyse with LBM:
``` python
    from sparsebm import generate_bernouilli_LBM
    import numpy as np
    #
    # Define the properties of your graph
    #
    number_of_rows = 10 ** 3
    number_of_columns = 2* number_of_rows
    nb_row_clusters, nb_column_clusters = 3, 4
    row_cluster_proportions = np.ones(nb_row_clusters) / nb_row_clusters
    column_cluster_proportions = np.ones(nb_column_clusters) / nb_column_clusters
    connection_probabilities = np.array([[0.1, 0.0125, 0.0125, 0.05], [0.0125, 0.1, 0.0125, 0.05], [0, 0.0125, 0.1, 0]])
    #
    # The graph is generated
    #
    data = generate_bernouilli_LBM(
            number_of_rows,
            number_of_columns,
            nb_row_clusters,
            nb_column_clusters,
            connection_probabilities,
            row_cluster_proportions,
            column_cluster_proportions
        )
    graph, row_cluster_indicator, column_cluster_indicator, = (data["X"], data["Y1"], data["Y2"])
```

### Infere with sparsebm LBM_bernouilli:
 - Use the bernouilli Latent Bloc Model:

``` python
    from sparsebm import LBM_bernouilli

    model = LBM_bernouilli(
        nb_row_clusters,
        nb_column_clusters,
        gpu_number=None, # Or give the desired GPU index.
    )
    model.fit(graph)

    print("Row labels:")
    print(model.row_labels)

    print("Column labels:")
    print(model.column_labels)
```
To use GPU acceleration, CUPY needs to be installed and set gpu_number the desired GPU index.
