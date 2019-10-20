 # Concrete definition
"""
    HeapKNNGraph{V, K, U}

A weighted, directed graph representing an approximate k-nearest neighbors graph
using binary max heaps to store each vertex's forward edges, allowing for
efficient updates of the candidate neighbors.
"""
struct HeapKNNGraph{V, K, U<:Real} <: ApproximateKNNGraph{V, K, U}
    _knn_heaps::Vector{BinaryMaxHeap}
end
# constructors TODO
function HeapKNNGraph(data::D, k::Integer, metric::PreMetric) where {D <: AbstractVector}
    # assert some invariants
    k > 0 || error("HeapKNNGraph needs positive `k`")
    length(data) > k || error("HeapKNNGraph needs `length(data) > k`")

    np = length(data)
    U = result_type(metric, data[1], data[1])
    HeapType = BinaryMaxHeap{HeapKNNGraphEdge{typeof(k), U}}
    knn_heaps = HeapType[HeapType() for _ in 1:np]
    for i in 1:np
        k_idxs = sample_neighbors(np, k, exclude=[i])
        for j in k_idxs
            dist = evaluate(metric, data[i], data[j])
            # calculate reverese distance if dist measure is not symmetric
            rev_dist = typeof(metric) <: SemiMetric ? dist : evaluate(metric, data[j], data[i])
            _heappush!(knn_heaps[i], HeapKNNGraphEdge(i, j, dist), k)
            _heappush!(knn_heaps[j], HeapKNNGraphEdge(j, i, rev_dist), k)
        end
    end
    return HeapKNNGraph{typeof(k), k, U}(knn_heaps)
end

# lightgraphs interface
"""
    edges(g::HeapKNNGraph)

Return an iterator (or collection(?)) of the edges in this graph.
"""
function edges(g::HeapKNNGraph) end

function vertices(g::HeapKNNGraph) end

function weights(g::HeapKNNGraph{V, K, U})::AbstractMatrix{U} where {V, K, U} end

function edgetype(g::HeapKNNGraph{V, K, U}) where {V, K, U} end

function has_edge(g::HeapKNNGraph{V}, s, d) where V end
function has_edge(g::HeapKNNGraph{V, K, U}, edge) where {V, K, U} end

function has_vertex(g::HeapKNNGraph{V}, v) where V end

function ne(g::HeapKNNGraph) end

function nv(g::HeapKNNGraph) end

"""
    inneighbors(g::HeapKNNGraph, v)

Return all neighbors connected to `v` by an incoming edge.
"""
function inneighbors(g::HeapKNNGraph, v) end

function outneighbors(g::HeapKNNGraph, v) end

# nndescent utilities

"""
    neighbors(g::HeapKNNGraph)

Return all the forward and reverse neighbors for every vertex in `g`.
"""
function neighbors(g::HeapKNNGraph) end
"""
    neighbors(g::HeapKNNGraph, v)

Return all forward and reverse neighbors for vertex `v` in `g`.
"""
function neighbors(g::HeapKNNGraph, v) end

"""
    add_edge!(g::HeapKNNGraph, e::HeapKNNGraphEdge)

Try to add an edge `e`, indicating a new candidate nearest neighbor `e.dest` for
vertex `e.src`, by pushing onto `e.src`'s heap. This will fail (return `0`) if
`e` is already in the heap, if `e.weight > top(heap).weight`. Otherwise return `1`.
If `length(heap) > K` after pushing, `pop` the largest candidate.
"""
function add_edge!(g::HeapKNNGraph, e::HeapKNNGraphEdge)
    # NOTE we can assume the invariants for heap knn graphs hold
    if e < top(g._knn_heaps[src(e)]) && !has_edge(g, e)
        push!(g._knn_heaps[src(e)], e)
        pop!(g._knn_heaps[src(e)])
        return 1
    end
    return 0
end
