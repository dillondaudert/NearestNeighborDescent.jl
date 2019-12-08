# Usage

## Graph Construction `nndescent`
`nndescent` builds the approximate kNN graph:
```julia
nndescent(data, n_neighbors, metric; max_iters, sample_rate, precision) -> graph
```
- `data`: The set of points to build the tree from. This must be of type
  `Vector{V}`, where `V <: AbstractVector` **or** `AbstractMatrix`.
- `n_neighbors`: An integer specifies the number of neighbors to find
- `metric`: Any metric `M` where `M <: PreMetric` from the Distances.jl package.

The performance of NN Descent can be tuned with several keyword arguments.
- `max_iters`: This controls the maximum number of iterations to search for
  neighbors. Default is `10`.
- `sample_rate`: The algorithm performs a local join around the candidate
  neighbors of each point during execution. The sample rate is the likelihood
  that each candidate be included in the local join for an iteration. Default is
  `1.`.
- `precision`: This argument roughly corresponds to the fraction of true
  nearest neighbors that will be missed by the algorithm. Default `.001`.

The k-nearest neighbors can be accessed with
```julia
knn_matrices(graph) -> indices, distances
```
The `indices` and `distances` are both `KxN` matrices containing ids and distances to each
point's neighbors, respectively, where `K = n_neighbors` and `N = length(data)`.

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

## Querying KNN Graphs `search`
Once constructed, the `ApproximateKNNGraph` can be used to find the nearest
neighbors to new points. This is done via the `search` method:
```julia
search(graph, queries, n_neighbors; max_candidates) -> indices, distances
```
- `graph`: An instance of `ApproximateKNNGraph`
- `queries`: A vector of new data points of type `Vector{V}` or `AbstractMatrix`.
  Note that the dimensionality of the queries must match that of the data used to
  originally construct the graph.
- `n_neighbors`: The number of neighbors to find for each query. This does
  *not* have to be the same as the number used to construct `graph`.
- `max_candidates`: Each query maintains a heap of candidate neighbors - the
  maximum size of that heap is controlled with this keyword arg (default `max(n_neighbors, 20)`).


Similar to `knn_matrices`, this returns two matrices for the indices and
distances to the nearest neighbors of each query.

Example:
```julia
queries = [rand(10) for _ in 1:100]
# OR queries = rand(10, 100)
indices, distances = search(graph, queries, 5)
```
