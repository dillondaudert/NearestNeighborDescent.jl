# Approximate *k*-Nearest Neighbor Graphs
The index structure built by NNDescent is an approximate *k*-NN graph.

**Definition**: An approximate k-nearest neighbor graph is a weighted, directed
graph *G = {V, E, k}*, where *V* is a non-empty set of nodes, *E* is a set of
weighted directed edges *(p, q, w)*, and *k* is a positive integer, with the
following properties:
1. | *V* | >  *k*
2. Every node *v* in *V* has exactly *k* outgoing edges, with no self-loops.

Note this implies that | *E* | = | *V* | * *k*. The edges *(p, q)* in *E* denote that
*q* is an *approximate* (or candidate) nearest neighbor of *p* in *V*.

For the types in this package, the set of nodes is immutable but the set of
outgoing edges for each node can be updated such that the sum of all outgoing
edge weights never increases (with `add_edge!`).

## Types

`ApproximateKNNGraph <: AbstractGraph`: An abstract type whose subtypes
all have the properties of approximate knn graphs

`HeapKNNGraph <: ApproximateKNNGraph`: A knn graph where each node's
outgoing edges are stored in a max heap

`ThreadSafeHeapKNNGraph`: A thread-safe version of `HeapKNNGraph`. This is
mostly for demonstration purposes, as (at writing) little or no speedup is observed
when using this struct with multiple threads.
