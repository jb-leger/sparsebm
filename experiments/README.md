# Experiments

## Analyzing the benefit of the inference optimized for sparse graphs.

The computational benefit of using the right inference is analyzed by comparing the inference of the block models optimized for sparse graph with the standard one.
Experiments are conducted on the latent block model only, but similar results can be expected for the stochastic block model.

### Fixed scarcity of edges, network size varying
:warning: **This experiment must be run with a GPU. Use google colab is you don't have one.**


1. Generate the datasets with
```
python experiments/sparse_sparsity_fixed_generate_data.py
```
If you want to experiment on more or less repetitions use argument repeat:
```
python experiments/sparse_sparsity_fixed_generate_data.py --repeat 1
```

2. The LBM is trained on each data matrix using the inference optimized for sparse graph and the original one.
To launch the inference use:
```
python experiments/sparse_sparsity_fixed.py
```

3. To plot the results, use:
```
python experiments/sparse_sparsity_fixed_results.py
```


### Fixed network size, scarcity of edges varying

With this experiment, we show that SparseBM implementation preserves the GPU memory and is faster when the sparsity rate is low.

:warning: This experiment **must** be run with a **GPU with at least 20GB** of memory.

1. Generate the datasets with
```
python experiments/sparse_size_fixed_generate_data.py
```
If you want to experiment on more or less repetitions use argument repeat:
```
python experiments/sparse_size_fixed_generate_data.py --repeat 1
```

2. The LBM is trained on each data matrix using the inference optimized for sparse graph and the original one.
To launch the inference use:
```
python experiments/sparse_size_fixed.py
```

3. To plot the results, use:
```
python experiments/sparse_size_fixed_results.py
```


## Comparing SparseBM with existing R packages.

1. The implementation of the LBM from SparseBM is compared with the ones available in the packages Blockcluster and Blockmodels.
The data set generated in the previous experiment on the growing network size and fixed sparsity rate is re-used. If the dataset has not been generated yet, use:
```
python experiments/sparse_sparsity_fixed_generate_data.py
```
If you want to experiment on more or less repetitions use argument repeat:
```
python experiments/sparse_sparsity_fixed_generate_data.py --repeat 1
```

2. :warning: To execute this experiment, **[R must be installed](https://cran.r-project.org/bin/linux/ubuntu/README.html)** as well as the **[blockcluster](https://cran.r-project.org/web/packages/blockcluster/index.html)** and **[blockmodels](https://cran.r-project.org/web/packages/blockmodels/index.html)** packages.
To bind R packages with Python, the rpy2 module must be installed with
```
pip install rpy2
```

3. To launch the benchmark with all algorithms, use:
```
python experiments/benchmark_libraries.py
```
To avoid memory problems, the programs gets the system memory and adapts, for each algorithm, the maximum size of the matrix to work with.
To bypass  this default behaviour, you could specify the maximum size and the algorithm you want to use with:
```
python experiments/benchmark_libraries.py --programs sparsebm --size=80000
```
In that case the benchmark is run with only SparseBM and on all matrices with maximum shapes 80000 times 40000.

4. To plot the results, use:
```
python experiments/benchmark_libraries_results.py
```
