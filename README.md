# NearestNeighborDescent.jl

| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] [![][coveralls-img]][coveralls-url] |

A Julia implementation of Nearest Neighbor Descent.

> Dong, Wei *et al.* Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures. *WWW* (2011).

## Overview

Nearest Neighbor Descent (NNDescent) is an approximate K-nearest neighbor graph construction algorithm that has
several useful properties:
- **general**: works with arbitrary dissimilarity functions
- **scalable**: has an empirical complexity of O(n^1.14) pairwise comparisons for a dataset of size n
- **space efficient**: the only data structure required is an approximate KNN graph which is operated on in-place and is also the final output
- **accurate**: converges to above 90% recall while only comparing each data point to a small percentage of the whole dataset on average

NNDescent is based on the heuristic argument that *a neighbor of a neighbor is also likely to be a neighbor*. That is,
given a list of approximate nearest neighbors to a point, we can improve that list by exploring the neighbors of each
point in the list. The algorithm is in essence the repeated application of this principle.

## Installation
```julia
]add NearestNeighborDescent
```

## Basic Usage

Approximate kNN graph construction on a dataset:

```julia
using NearestNeighborDescent
using Distances
data = [rand(20) for _ in 1:1000]
n_neighbors = 10
metric = Euclidean()
graph = nndescent(data, n_neighbors, metric)
```

The approximate KNNs of the original dataset can be retrieved from the resulting graph with
```julia
# return the approximate knns as KxN matrices of indexes and distances, where
# indices[j, i] and distances[j, i] are the index of and distance to node i's jth
# nearest neighbor, respectively.
indices, distances = knn_matrices(graph)
```

To find the approximate neighbors for new points with respect to an already constructed graph:

```julia
queries = [rand(20) for _ in 1:20]
n_neighbors = 5
indices, distances = search(graph, queries, n_neighbors)
```

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://dillondaudert.github.io/NearestNeighborDescent.jl/stable

[travis-img]: https://travis-ci.com/dillondaudert/NearestNeighborDescent.jl.svg?branch=master
[travis-url]: https://travis-ci.com/dillondaudert/NearestNeighborDescent.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/lr49p9vxkr8a3uv0?svg=true
[appveyor-url]: https://ci.appveyor.com/project/dillondaudert/nearestneighbordescent-jl

[codecov-img]: https://codecov.io/gh/dillondaudert/NearestNeighborDescent.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/dillondaudert/NearestNeighborDescent.jl

[coveralls-img]: https://coveralls.io/repos/github/dillondaudert/NearestNeighborDescent.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/dillondaudert/NearestNeighborDescent.jl?branch=master
