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

Return an iterator of the edges in this KNN graph.
"""
function LightGraphs.edges(g::HeapKNNGraph)
    return (e for heap in g._knn_heaps for e in heap.valtree)
end

"""
    vertices(g::HeapKNNGraph)

Return an iterator of the vertices in the KNN graph.
"""
LightGraphs.vertices(g::HeapKNNGraph{V}) where V = one(V):V(length(g._knn_heaps))

"""
    weights(g::HeapKNNGraph{V, K, U})

Returns a SparseMatrixCSC{U, V} `w` where `w[j, i]` returns the weight of the
directed edge from vertex `i` to `j`.
"""
function LightGraphs.weights(g::HeapKNNGraph{V, K, U}) where {V, K, U}
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
function LightGraphs.edgetype(g::HeapKNNGraph{V, K, U}) where {V, K, U}
    return HeapKNNGraphEdge{V, U}
end

"""
    has_edge(g::HeapKNNGraph, s, d)

Return true if the graph g has an edge from `s` to `d`, else false.
"""
function LightGraphs.has_edge(g::HeapKNNGraph{V}, s, d) where V
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
function LightGraphs.has_edge(g::HeapKNNGraph{V, K, U}, e::HeapKNNGraphEdge{V, U}) where {V, K, U}
    return e in g._knn_heaps[src(e)].valtree
end

"""
    has_vertex(g::HeapKNNGraph, v)

Return true if vertex `v` is in `g`.
"""
LightGraphs.has_vertex(g::HeapKNNGraph{V}, v::V) where V = v in 1:nv(g)

"""
    ne(g::HeapKNNGraph)

Return the number of edges in `g`.
"""
LightGraphs.ne(g::HeapKNNGraph{V, K, U}) where {V, K, U} = K*nv(g)

"""
    nv(g::HeapKNNGraph)

Return the number of vertices in `g`.
"""
LightGraphs.nv(g::HeapKNNGraph) = length(g._knn_heaps)

"""
    inneighbors(g::HeapKNNGraph, v)

Return a list of the neighbors connected to `v` by an incoming edge.

**Implementation Notes**

HeapKNNGraph doesn't store inneighbors directly; it must find them by iterating
over the outgoing edges for each vertex and saving those where `v == dst(e)`.
Thus, this has time complexity `𝒪(nv(g)*K)`.
"""
function LightGraphs.inneighbors(g::HeapKNNGraph{V}, v::V) where V
    return collect(src(e) for e in edges(g) if dst(e) == v)
end

"""
    outneighbors(g::HeapKNNGraph, v)

Return a list of the neighbors of `v` connected by outgoing edges.

**Implementation Notes**

HeapKNNGraph stores each vertex's outgoing edges in a heap, so this has a time
complexity of `𝒪(K)`.
"""
LightGraphs.outneighbors(g::HeapKNNGraph{V}, v::V) where V = dst.(g._knn_heaps[v].valtree)

"""
    add_edge!(g::HeapKNNGraph, e::HeapKNNGraphEdge)

Try to add an edge `e`, indicating a new candidate nearest neighbor `e.dest` for
vertex `e.src`, by pushing onto `e.src`'s heap. This will fail (return `0`) if
`e` is already in the heap, if `e.weight > top(heap).weight`. Otherwise return `1`.
If `length(heap) > K` after pushing, `pop` the largest candidate.
"""
function LightGraphs.add_edge!(g::HeapKNNGraph, e::HeapKNNGraphEdge)
    # NOTE we can assume the invariants for heap knn graphs hold
    if e < top(g._knn_heaps[src(e)]) && !has_edge(g, e)
        push!(g._knn_heaps[src(e)], e)
        pop!(g._knn_heaps[src(e)])
        return 1
    end
    return 0
end

# KNNGraphs interface methods
function knn_diameter(g::HeapKNNGraph{V}, v::V) where V
    return 2 * weight(top(g._knn_heaps[v]))
end
