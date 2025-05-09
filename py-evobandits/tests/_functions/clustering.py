# Copyright 2025 EvoBandits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Objective function and useful parameters for a clustering problem,
which serves as an example to demonstrate the use of various parameters types.

The test example is based on an example from sklearn:
Title: Comparison of the K-Means and MiniBatchKMeans clustering algorithms
Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#sphx-glr-auto-examples-cluster-plot-mini-batch-kmeans-py
Last accessed: 2025-04-18
Version: 1.6.1
"""

# import pandas as pd
import numpy as np
from evobandits import Arm, CategoricalParam, FloatParam, IntParam
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs

# Useful parameters
PARAMS = {
    "algorithm": CategoricalParam([KMeans, MiniBatchKMeans]),
    "init": CategoricalParam(["k-means++", "random"]),
    "n_clusters": IntParam(1, 10),
    "tol": FloatParam(1e-4, 1e-2),
}
BOUNDS = [(0, 1), (0, 1), (1, 10), (0, 100)]
RESULTS_EXAMPLE = [Arm([0, 0, 4, 0]), Arm([0, 0, 3, 0])]
BEST_TRIAL_EXAMPLE = [
    {
        "n_best": 1,
        "mean_reward": 0.0,
        "num_pulls": 0,
        "params": {
            "algorithm": KMeans,
            "init": "k-means++",
            "n_clusters": 4,
            "tol": 0.0001,
        },
    },
    {
        "n_best": 2,
        "mean_reward": 0.0,
        "num_pulls": 0,
        "params": {
            "algorithm": KMeans,
            "init": "k-means++",
            "n_clusters": 3,
            "tol": 0.0001,
        },
    },
]
# TRIALS_DF = pd.DataFrame(BEST_TRIAL_EXAMPLE)


# Generate sample data
np.random.seed(0)

_centers = [[1, 1], [-1, -1], [1, -1]]
_n_clusters = len(_centers)
_X, labels_true = make_blobs(n_samples=10000, centers=_centers, cluster_std=0.7)


def function(algorithm, init, n_clusters, tol) -> float:
    """Evaluate the inertia of the clustering that results from the given parameters."""
    clusterer = algorithm(init=init, n_clusters=n_clusters, tol=tol, n_init=10)
    clusterer.fit(_X)
    return clusterer.inertia_


if __name__ == "__main__":
    # Example usage
    result = function(KMeans, "k-means++", 3, 0.001)
    print(f"Clustering inertia: {result}")
