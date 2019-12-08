# NearestNeighborDescent.jl

[![Build Status](https://travis-ci.com/dillondaudert/NearestNeighborDescent.jl.svg?branch=master)](https://travis-ci.com/dillondaudert/NearestNeighborDescent.jl) [![Build status](https://ci.appveyor.com/api/projects/status/lr49p9vxkr8a3uv0?svg=true)](https://ci.appveyor.com/project/dillondaudert/nearestneighbordescent-jl)
 [![codecov](https://codecov.io/gh/dillondaudert/NearestNeighborDescent.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dillondaudert/NearestNeighborDescent.jl) [![Coverage Status](https://coveralls.io/repos/github/dillondaudert/NearestNeighborDescent.jl/badge.svg?branch=master)](https://coveralls.io/github/dillondaudert/NearestNeighborDescent.jl?branch=master)[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://dillondaudert.github.io/NearestNeighborDescent.jl/dev)

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

## Basic Usage

One-shot approximate kNN graph construction on a dataset:

```julia
nndescent(data, n_neighbors, metric; max_iters, sample_rate, precision) -> graph
```

The approximate KNNs of the original dataset can be retrieved from the resulting graph with
```julia
# return the approximate knns as matrices of indexes and distances, where
# indices[j, i] and distances[j, i] are the index of and distance to node i's jth
# nearest neighbor, respectively.
knn_matrices(graph) -> indices, distances
```

To find the approximate neighbors for new points with respect to an already constructed graph:

```julia
search(graph, queries, n_neighbors) -> indices, distances
```

`graph` is a `ApproximateKNNGraph <: AbstractGraph`, and all the usual `LightGraphs` utilities
will work on it. 
