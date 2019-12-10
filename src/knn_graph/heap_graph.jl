# Concrete definition
"""
    HeapKNNGraph{V, U, D, M}

A weighted, directed graph representing an approximate k-nearest neighbors graph
using binary max heaps to store each vertex's forward edges, allowing for
efficient updates of the candidate neighbors.
"""
struct HeapKNNGraph{V<:Integer,
                    U<:Real,
                    D<:AbstractVector,
                    M<:PreMetric} <: ApproximateKNNGraph{V, U, D, M}
    heaps::Vector{BinaryMaxHeap{HeapKNNGraphEdge{V, U}}}
    data::D
    n_neighbors::V
    metric::M
end

"""
    HeapKNNGraph(data, metric, indices, distances)

Create a HeapKNNGraph from `KxN` matrices of the indices and distances, where `indices[i, v]` and
`distances[i, v]` are the index and distance to node `v`s `i`th candidate neighbor. Note that each
column of `indices` cannot have duplicate entries, but they need not be sorted by distance.
"""
function HeapKNNGraph(data::D,
                      metric::M,
                      indices::AbstractMatrix{V},
                      distances::AbstractMatrix{U}) where {D <: AbstractVector,
                                                           M <: PreMetric,
                                                           V <: Integer,
                                                           U <: Real}
    n_neighbors = size(indices, 1)
    n_points = size(indices, 2)
    length(data) == size(indices, 2) || error("`indices` must have num columns equal to `length(data)`")
    size(indices) == size(distances) || error("`indices` and `distances` must have same shape")
    n_neighbors < n_points || error("`Must have more columns than rows`")
    # ...
    HeapType = BinaryMaxHeap{HeapKNNGraphEdge{V, U}}
    knn_heaps = HeapType[HeapType() for _ in 1:n_points]
    for v in 1:n_points, i in 1:n_neighbors
        _heappush!(knn_heaps[v], HeapKNNGraphEdge(v, indices[i, v], distances[i, v]), n_neighbors)
    end
    return HeapKNNGraph{V, U, D, M}(knn_heaps, data, n_neighbors, metric)
end

"""
    HeapKNNGraph(data, n_neighbors, metric)

Create a HeapKNNGraph by randomly sampling `n_neighbors` for each point and using `metric`
to calculate weights.
"""
function HeapKNNGraph(data::D, n_neighbors::Integer, metric::M) where {D <: AbstractVector,
                                                                       M <: PreMetric}
    # assert some invariants
    n_neighbors > 0 || error("HeapKNNGraph needs positive `n_neighbors`")
    length(data) > n_neighbors || error("HeapKNNGraph needs `length(data) > n_neighbors`")

    n_points = length(data)
    U = result_type(metric, data[1], data[1])
    HeapType = BinaryMaxHeap{HeapKNNGraphEdge{typeof(n_neighbors), U}}
    knn_heaps = HeapType[HeapType() for _ in 1:n_points]
    # create approximate knn heaps for each point
    for i in eachindex(data)
        k_idxs = sample_neighbors(n_points, n_neighbors, exclude=[i])
        for j in k_idxs
            dist = evaluate(metric, data[i], data[j])
            # calculate reverese distance if dist measure is not symmetric
            rev_dist = metric isa SemiMetric ? dist : evaluate(metric, data[j], data[i])
            _heappush!(knn_heaps[i], HeapKNNGraphEdge(i, j, dist), n_neighbors)
            _heappush!(knn_heaps[j], HeapKNNGraphEdge(j, i, rev_dist), n_neighbors)
        end
    end
    return HeapKNNGraph{typeof(n_neighbors), U, D, M}(knn_heaps, data, n_neighbors, metric)
end

# lightgraphs interface
"""
    edges(g::HeapKNNGraph)

Return an iterator of the edges in this KNN graph. Note that mutating the edges
while iterating through `edges(g)` may result in undefined behavior.
"""
@inline function LightGraphs.edges(g::Union{HeapKNNGraph, LockHeapKNNGraph})
    return (e for heap in g.heaps for e in heap.valtree)
end

"""
    vertices(g::HeapKNNGraph)

Return an iterator of the vertices in the KNN graph.
"""
@inline function LightGraphs.vertices(g::Union{HeapKNNGraph{V}, LockHeapKNNGraph{V}}) where V
    return one(V):V(length(g.heaps))
end

"""
    weights(g::HeapKNNGraph{V, U})

Returns a SparseMatrixCSC{U, V} `w` where `w[j, i]` returns the weight of the
directed edge from vertex `i` to `j`.
"""
function LightGraphs.weights(g::Union{HeapKNNGraph{V, U}, LockHeapKNNGraph{V, U}}) where {V, U}
    # dests, srcs, ws where weights[dests[k], srcs[k]] = ws[k]
    srcs, dsts = V[], V[]
    vals = U[]
    for e in edges(g)
        push!(srcs, src(e))
        push!(dsts, dst(e))
        push!(vals, weight(e))
    end
    return sparse(dsts, srcs, vals, nv(g), nv(g))
end

"""
    edgetype(g::HeapKNNGraph)

Return the type of the edges in `g`.
"""
function LightGraphs.edgetype(g::Union{HeapKNNGraph{V, U}, LockHeapKNNGraph{V, U}}) where {V, U}
    return HeapKNNGraphEdge{V, U}
end

"""
    has_edge(g::HeapKNNGraph, s, d)

Return true if the graph g has an edge from `s` to `d`, else false.
"""
@inline function LightGraphs.has_edge(g::Union{HeapKNNGraph{V}, LockHeapKNNGraph{V}}, s, d) where V
    for e in g.heaps[s].valtree
        if dst(e) == d
            return true
        end
    end
    return false
end
"""
    has_edge(g::HeapKNNGraph, e::HeapKNNGraphEdge)

Return true if `g` contains the edge `e`.
"""
@inline function LightGraphs.has_edge(g::Union{HeapKNNGraph{V, U}, LockHeapKNNGraph{V, U}},
                                      e::HeapKNNGraphEdge{V, U}) where {V, U}
    return e in g.heaps[src(e)].valtree
end

"""
    has_vertex(g::HeapKNNGraph, v)

Return true if vertex `v` is in `g`.
"""
@inline LightGraphs.has_vertex(g::Union{HeapKNNGraph{V}, LockHeapKNNGraph{V}}, v::V) where V = v in 1:nv(g)

"""
    ne(g::HeapKNNGraph)

Return the number of edges in `g`.
"""
@inline LightGraphs.ne(g::Union{HeapKNNGraph{V, U}, LockHeapKNNGraph{V, U}}) where {V, U} = g.n_neighbors*nv(g)

"""
    nv(g::HeapKNNGraph)

Return the number of vertices in `g`.
"""
@inline LightGraphs.nv(g::Union{HeapKNNGraph, LockHeapKNNGraph}) = length(g.heaps)

"""
    inneighbors(g::HeapKNNGraph, v)

Return a list of the neighbors connected to `v` by an incoming edge.

**Implementation Notes**

HeapKNNGraph doesn't store inneighbors directly; it must find them by iterating
over the outgoing edges for each vertex and saving those where `v == dst(e)`.
Thus, this has time complexity `𝒪(nv(g)*K)`.
"""
@inline function LightGraphs.inneighbors(g::HeapKNNGraph{V}, v::V) where V
    return collect(src(e) for e in edges(g) if dst(e) == v)
end

"""
    outneighbors(g::HeapKNNGraph, v)

Return a list of the neighbors of `v` connected by outgoing edges.

**Implementation Notes**

HeapKNNGraph stores each vertex's outgoing edges in a heap, so this has a time
complexity of `𝒪(K)`.
"""
@inline LightGraphs.outneighbors(g::HeapKNNGraph{V}, v::V) where V = dst.(g.heaps[v].valtree)

"""
    add_edge!(g::HeapKNNGraph, e::HeapKNNGraphEdge)

Try to add an edge `e`, indicating a new candidate nearest neighbor `e.dest` for
vertex `e.src`, by pushing onto `e.src`'s heap. This will fail (return `false`) if
`e` is already in the heap or if `e.weight > top(heap).weight`. Otherwise return `true`.
"""
function LightGraphs.add_edge!(g::HeapKNNGraph, e::HeapKNNGraphEdge)
    # NOTE we can assume the invariants for heap knn graphs hold
    if e < top(g.heaps[src(e)]) && !has_edge(g, src(e), dst(e))
        # we know this edge is smaller than the top, so we can start by removing that
        pop!(g.heaps[src(e)])
        push!(g.heaps[src(e)], e)
        return true
    end
    return false
end

# KNNGraphs public methods
"""
    knn_diameter(graph, v) -> diameter

Return the diameter of the set of KNNs of vertex `v`.
"""
function knn_diameter(g::Union{HeapKNNGraph{V}, LockHeapKNNGraph{V}}, v::V) where V
    return 2 * weight(top(g.heaps[v]))
end

"""
    knn_matrices(graph) -> indices, distances

Return the indices and distances of the approximate KNNs as dense
matrices where indices[j, i] and distances[j, i] are the index of
and distance to node i's jth nearest neighbor, respectively.
"""
function knn_matrices(g::Union{HeapKNNGraph{V, U}, LockHeapKNNGraph{V, U}}) where {V, U}
    indices = Matrix{V}(undef, g.n_neighbors, nv(g))
    distances = Matrix{U}(undef, g.n_neighbors, nv(g))
    for i in vertices(g)
        i_edges = sort!(node_edges(g, i))
        for (j, e) in enumerate(i_edges)
            indices[j, i] = dst(e)
            distances[j, i] = weight(e)
        end
    end
    return indices, distances
end

"""
    edge_indices(graph) -> CartesianIndices

Return the indices of the KNNs for each node `v` as tuples (v, i). To be used with
`node_edge(graph, v, i)`.
"""
function edge_indices(g::ApproximateKNNGraph)
    return CartesianIndices((nv(g), g.n_neighbors))
end

"""
    node_edge(graph, v, i) -> edge

Return the ith outgoing edge from node `v`. No ordering of the
edges is guaranteed; in particular, node_edge(graph, v, 1) is not
guaranteed to be the edge to v's nearest neighbor.
"""
@inline function node_edge(g::Union{HeapKNNGraph{V}, LockHeapKNNGraph{V}}, v::V, i::V) where V
    return g.heaps[v].valtree[i]
end

"""
    node_edges(graph, v) -> edges

Return all the outgoing edges from node v in an arbitrary order.
"""
@inline function node_edges(g::HeapKNNGraph{V}, v::V) where V
    return edgetype(g)[node_edge(g, v, i) for i in one(V):g.n_neighbors]
end

"""
    update_flag!(graph, v, i, new_flag)

Update the flag of node `v`s ith outgoing edge. Returns the new edge.
Note that since the flags don't influence the edge ordering, this can't invalidate the heap
invariant.
"""
function update_flag!(g::HeapKNNGraph{V}, v::V, i::V, new_flag::Bool) where V
    edge = node_edge(g, v, i)
    new_edge = edgetype(g)(src(edge), dst(edge), weight(edge), new_flag)
    g.heaps[v].valtree[i] = new_edge
    return new_edge
end
