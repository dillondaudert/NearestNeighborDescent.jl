# Concrete definition
"""
    HeapKNNGraph{V, K, U}

A weighted, directed graph representing an approximate k-nearest neighbors graph
using binary max heaps to store each vertex's forward edges, allowing for
efficient updates of the candidate neighbors.
"""
struct HeapKNNGraph{V<:Integer, K, U<:Real} <: ApproximateKNNGraph{V, K, U}
    _knn_heaps::Vector{BinaryMaxHeap{HeapKNNGraphEdge{V, U}}}
end
function HeapKNNGraph(data::D, k::Integer, metric::PreMetric) where {D <: AbstractVector}
    # assert some invariants
    k > 0 || error("HeapKNNGraph needs positive `k`")
    length(data) > k || error("HeapKNNGraph needs `length(data) > k`")

    np = length(data)
    U = result_type(metric, data[1], data[1])
    HeapType = BinaryMaxHeap{HeapKNNGraphEdge{typeof(k), U}}
    knn_heaps = HeapType[HeapType() for _ in 1:np]
    # create approximate knn heaps for each point
    for i in eachindex(data)
        k_idxs = sample_neighbors(np, k, exclude=[i])
        for j in k_idxs
            dist = evaluate(metric, data[i], data[j])
            # calculate reverese distance if dist measure is not symmetric
            rev_dist = metric isa SemiMetric ? dist : evaluate(metric, data[j], data[i])
            _heappush!(knn_heaps[i], HeapKNNGraphEdge(i, j, dist), k)
            _heappush!(knn_heaps[j], HeapKNNGraphEdge(j, i, rev_dist), k)
        end
    end
    return HeapKNNGraph{typeof(k), k, U}(knn_heaps)
end

# lightgraphs interface
"""
    edges(g::HeapKNNGraph)

Return an iterator of the edges in this KNN graph. Note that mutating the edges
while iterating through `edges(g)` may result in undefined behavior.
"""
@inline function LightGraphs.edges(g::Union{HeapKNNGraph, LockHeapKNNGraph})
    return (e for heap in g._knn_heaps for e in heap.valtree)
end

"""
    vertices(g::HeapKNNGraph)

Return an iterator of the vertices in the KNN graph.
"""
@inline function LightGraphs.vertices(g::Union{HeapKNNGraph{V}, LockHeapKNNGraph{V}}) where V
    return one(V):V(length(g._knn_heaps))
end

"""
    weights(g::HeapKNNGraph{V, K, U})

Returns a SparseMatrixCSC{U, V} `w` where `w[j, i]` returns the weight of the
directed edge from vertex `i` to `j`.
"""
function LightGraphs.weights(g::Union{HeapKNNGraph{V, K, U}, LockHeapKNNGraph{V, K, U}}) where {V, K, U}
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
function LightGraphs.edgetype(g::Union{HeapKNNGraph{V, K, U}, LockHeapKNNGraph{V, K, U}}) where {V, K, U}
    return HeapKNNGraphEdge{V, U}
end

"""
    has_edge(g::HeapKNNGraph, s, d)

Return true if the graph g has an edge from `s` to `d`, else false.
"""
@inline function LightGraphs.has_edge(g::Union{HeapKNNGraph{V}, LockHeapKNNGraph{V}}, s, d) where V
    for e in g._knn_heaps[s].valtree
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
@inline function LightGraphs.has_edge(g::Union{HeapKNNGraph{V, K, U}, LockHeapKNNGraph{V, K, U}},
                                      e::HeapKNNGraphEdge{V, U}) where {V, K, U}
    return e in g._knn_heaps[src(e)].valtree
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
@inline LightGraphs.ne(g::Union{HeapKNNGraph{V, K, U}, LockHeapKNNGraph{V, K, U}}) where {V, K, U} = K*nv(g)

"""
    nv(g::HeapKNNGraph)

Return the number of vertices in `g`.
"""
@inline LightGraphs.nv(g::Union{HeapKNNGraph, LockHeapKNNGraph}) = length(g._knn_heaps)

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
@inline LightGraphs.outneighbors(g::HeapKNNGraph{V}, v::V) where V = dst.(g._knn_heaps[v].valtree)

"""
    add_edge!(g::HeapKNNGraph, e::HeapKNNGraphEdge)

Try to add an edge `e`, indicating a new candidate nearest neighbor `e.dest` for
vertex `e.src`, by pushing onto `e.src`'s heap. This will fail (return `false`) if
`e` is already in the heap or if `e.weight > top(heap).weight`. Otherwise return `true`.
"""
function LightGraphs.add_edge!(g::HeapKNNGraph, e::HeapKNNGraphEdge)
    # NOTE we can assume the invariants for heap knn graphs hold
    if e < top(g._knn_heaps[src(e)]) && !has_edge(g, src(e), dst(e))
        # we know this edge is smaller than the top, so we can start by removing that
        pop!(g._knn_heaps[src(e)])
        push!(g._knn_heaps[src(e)], e)
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
    return 2 * weight(top(g._knn_heaps[v]))
end

"""
    knn_matrices(graph) -> indices, distances

Return the indices and distances of the approximate KNNs as dense
matrices where indices[j, i] and distances[j, i] are the index of
and distance to node i's jth nearest neighbor, respectively.
"""
function knn_matrices(g::Union{HeapKNNGraph{V, K, U}, LockHeapKNNGraph{V, K, U}}) where {V, K, U}
    indices = Matrix{V}(undef, K, nv(g))
    distances = Matrix{U}(undef, K, nv(g))
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

Return the indices of the KNNs (i, j). Can be used with
`node_edge(graph, i, j)`.
"""
function edge_indices(g::ApproximateKNNGraph{V, K}) where {V, K}
    return CartesianIndices((nv(g), K))
end

"""
    node_edge(graph, i, j) -> edge

Return the jth outgoing edge from node i. No ordering of the
edges is guaranteed; in particular, node_edge(graph, i, 1) is not
guaranteed to be the edge to i's nearest neighbor.
"""
@inline function node_edge(g::Union{HeapKNNGraph{V}, LockHeapKNNGraph{V}}, i::V, j::V) where V
    return g._knn_heaps[i].valtree[j]
end

"""
    node_edges(graph, i) -> edges

Return all the outgoing edges from node i in an arbitrary order.
"""
@inline function node_edges(g::HeapKNNGraph{V, K}, i::V) where {V, K}
    return edgetype(g)[node_edge(g, i, j) for j in one(V):K]
end

"""
Update the flag of the edge at the given indices. Since the flags don't influence
the edge ordering, this can't invalidate the heap invariant.
"""
function update_flag!(g::HeapKNNGraph{V}, i::V, j::V, flag::Bool) where V
    edge = g._knn_heaps[i].valtree[j]
    newedge = edgetype(g)(src(edge), dst(edge), weight(edge), flag)
    g._knn_heaps[i].valtree[j] = newedge
    return newedge
end
