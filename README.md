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
from sparsebm import generate_bernouilli_SBM
import numpy as np
from scipy import optimize
number_of_nodes = 10 ** 3
number_of_clusters = 4
# Define the properties of your graph
cluster_proportions = np.ones(number_of_clusters) / number_of_clusters
degree_wanted = 20
connection_probabilities = np.array([[8, 3, 1, 5], [3, 6, 0, 0], [1, 0, 9, 2], [5, 0, 2, 7]])
# The connection_probabilities are adapted to get the desired node degree
def f(x, number_of_nodes, pi, degree_wanted):
    return np.abs((connection_probabilities / (x * number_of_nodes)).mean() * number_of_nodes - degree_wanted)
res = optimize.minimize_scalar(lambda x: f(x, number_of_nodes, connection_probabilities, degree_wanted))
connection_probabilities = connection_probabilities / (res.x * number_of_nodes)
# The graph is generated
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

    # Define the properties of your graph
    number_of_rows = 10 ** 3
    number_of_columns = 2* number_of_columns
    nb_row_clusters, nb_column_clusters = 3, 4
    row_cluster_proportions = np.ones(nq) / nq
    column_cluster_proportions = np.ones(nl) / nl
    connection_probabilities = np.array([[8, 1, 1, 4], [1, 8, 1, 4], [0, 1, 8, 0]]) / (0.08 * number_of_rows)

    # The graph is generated
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
