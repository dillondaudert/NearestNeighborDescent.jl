# Usage

## Graph Construction
`nndescent` builds the approximate kNN graph:
```julia
nndescent(data, n_neighbors, metric; max_iters, sample_rate, precision) -> graph
```
- `data`: The set of points to build the graph from. This must either be a
  vector of points `AbstractVector{V}` or an `AbstractMatrix`. In the
  matrix case, each column is assumed to be a point.
- `n_neighbors`: A positive integer specifying the number of neighbors to find. As the
  quality of the approximate neighbors returned depends on this parameter, this should
  probably never be less than `5`.
- `metric`: Any subtype of the `PreMetric` abstract type from `Distances.jl`,
  including user-defined subtypes.

The behavior of NNDescent can be tuned with several keyword arguments.
- `max_iters`: This controls the maximum number of iterations to search for
  neighbors. Default is `10`.
- `sample_rate`: Controls how many candidate neighbors are evaluated each
  iteration. Lower values result in fewer distance calculations per iteration,
  trading off accuracy for speed. Must be a value in (0, 1]; default is `1`.
- `precision`: Execution will terminate early (i.e. before `max_iters` is reached)
  if the proportion of edges updated after an iteration is less than this value.
  Default `.001`.

The k-nearest neighbors can be accessed with
```julia
knn_matrices(graph) -> indices, distances
```
The `indices` and `distances` are both `KxN` matrices containing ids and distances to each point's neighbors, respectively, where `K = n_neighbors` and `N` is the number
of points in the dataset.

Example:
```julia
using NearestNeighborDescent
data = [rand(10) for _ in 1:1000]
# OR data = rand(10, 1000)

# nn descent search
graph = nndescent(data, 10, Euclidean())

# access point i's jth nearest neighbor:
indices, distances = knn_matrices(graph)
indices[j, i]
distances[j, i]
```

## Querying
Once constructed, the `ApproximateKNNGraph` can be used to find the nearest
neighbors to new points. This is done via the `search` method:
```julia
search(graph, queries, n_neighbors; max_candidates) -> indices, distances
```
- `graph`: An instance of `ApproximateKNNGraph`
- `queries`: A vector of new data points of type `Vector{V}` or `AbstractMatrix`.
  Note that the type of the query points must be compatible with the original
  dataset points.
- `n_neighbors`: The number of neighbors to find for each query. This does
  *not* have to be the same as the number used to construct `graph`.
- `max_candidates`: Each query maintains a heap of candidate neighbors - the
  maximum size of that heap is controlled with this keyword arg (default `max(n_neighbors, 20)`).

This will automatically use all available threads for the search.
Similar to `knn_matrices`, this returns two matrices for the indices and
distances to the nearest neighbors of each query.

Example:
```julia
queries = [rand(10) for _ in 1:100]
# OR queries = rand(10, 100)
indices, distances = search(graph, queries, 5)
```
