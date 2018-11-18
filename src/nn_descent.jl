# A simple NN graph implementation

struct DescentGraph{V <: AbstractVector,K,M,S <: AbstractVector}
    data::Vector{V}
    nneighbors::K
    metric::M
    graph::Vector{S}
end

"""
    DescentGraph(data, nneighbors [, metric = Euclidean()]) -> descentgraph
"""
function DescentGraph(data::Vector{V},
                     n_neighbors::Int,
                     metric::Metric = Euclidean()
                    ) where {V <: AbstractVector}
    graph = build_graph(data, metric, n_neighbors)
    DescentGraph(data, n_neighbors, metric, graph)
end

"""
    knn(graph::DescentGraph) -> ids, dists

Return the prebuilt kNN as a tuple `(ids, dists)` where `ids` is an `KxN` matrix
of integer indices and `dists` is an `KxN` matrix of distances.
"""
function knn(graph::DescentGraph)
    np, k = length(graph.graph), graph.nneighbors
    ids = Array{Int}(undef, (k, np))
    dists = Array{Float64}(undef, (k, np))

    for i = 1:np
        for j in 1:k
            ids[j, i] = graph.graph[i][j].idx
            dists[j, i] = graph.graph[i][j].dist
        end
    end
    return ids, dists
end

"""
Return a kNN graph for the input data according to the given metric.
"""
function build_graph(data::Vector{V},
                     metric::Metric,
                     k::Int,
                     sample_rate::R = 1.,
                     precision::R = .001
                    ) where {V <: AbstractArray, R <: AbstractFloat}

    np = length(data)
    # initialize with random neighbors
    knn_heaps = make_knn_heaps(data, k)

    # until no further updates
    while true
        # get the fw and bw neighbors of each point
        old_fw, fw, old_bw, bw = _neighbors(knn_heaps, sample_rate)
        old_neighbors = [union(old_fw[i], old_bw[i]) for i in 1:np]
        new_neighbors = [union(fw[i], bw[i]) for i in 1:np]
        c = 0
        # calculate local join around each point
        for i in 1:np
            for u₁ ∈ new_neighbors[i]
                for u₂ ∈ new_neighbors[i]
                    # both points are new
                    if i ≠ u₁ && i ≠ u₂ && u₁ < u₂
                        d = evaluate(metric, data[u₁], data[u₂])
                        c += _heappush!(knn_heaps[u₁], NNTuple(u₂, d))
                        c += _heappush!(knn_heaps[u₂], NNTuple(u₁, d))
                    end
                end
                for u₂ ∈ old_neighbors[i]
                    # one point is new
                    if i ≠ u₁ && i ≠ u₂
                        d = evaluate(metric, data[u₁], data[u₂])
                        c += _heappush!(knn_heaps[u₁], NNTuple(u₂, d))
                        c += _heappush!(knn_heaps[u₂], NNTuple(u₁, d))
                    end
                end
            end

        end
        if c < precision*k*np
            break
        end
    end
    knn_graph = [[pop!(knn_heaps[i]) for _ in 1:length(knn_heaps[i])][end-(k-1):end]
                    for i in 1:length(knn_heaps)]

    return knn_graph
end

"""
Get the neighbors of each point in a KNN graph, `graph`,
as an array of ids.
"""
function _neighbors(graph, sample_rate::AbstractFloat = 1.)
    old_fw_neighbors = [Vector{Int}() for _ in 1:length(graph)]
    new_fw_neighbors = [Vector{Int}() for _ in 1:length(graph)]
    old_bw_neighbors = [Vector{Int}() for _ in 1:length(graph)]
    new_bw_neighbors = [Vector{Int}() for _ in 1:length(graph)]
    for i in 1:length(graph), j in 1:length(graph[i])
        # add incoming edge ith -> jth.idx
        if rand() ≤ sample_rate
            if graph[i][j].flag
                # denote this neighbor has participated in local join
                graph[i][j].flag = false
                append!(new_fw_neighbors[i], graph[i][j].idx)
                append!(new_bw_neighbors[graph[i][j].idx], i)
            else
                append!(old_fw_neighbors[i], graph[i][j].idx)
                append!(old_bw_neighbors[graph[i][j].idx], i)
            end
        end
    end
    return old_fw_neighbors, new_fw_neighbors, old_bw_neighbors, new_bw_neighbors
end

"""
    search(graph::DescentGraph, queries::Vector{V}, n_neighbors, queue_size) -> indices, distances

Search the kNN `graph` for the nearest neighbors of the points in `queries`.
`queue_size` controls how large the candidate queue should be as a multiple of
`n_neighbors`. Larger values increase accuracy at the cost of speed, default=1.
"""
function search(graph::DescentGraph,
                queries::Vector{V},
                n_neighbors::Integer,
                queue_size::Real = 1.,
                ) where {V <: AbstractArray}
    max_candidates = trunc(Int, n_neighbors*queue_size)
    candidates = [binary_maxheap(NNTuple{Int, Float64}) for _ in 1:length(queries)]
    for i in eachindex(queries)
        # init
        j =  rand(1:length(graph.data))
        d = evaluate(graph.metric, queries[i], graph.data[j])
        _heappush!(candidates[i], NNTuple(j, d, false), max_candidates)

        while true
            unexp = unexpanded(candidates[i])
            if length(unexp) == 0
                break
            end
            # expand closest unexpanded neighbor
            unexp[1].flag = true
            #unexp[1].idx is idx in data of candidate neighbor to queries[i]
            # graph.graph[unexp[1].idx] is an array of NNtuples of the approx kNN
            for t in graph.graph[unexp[1].idx]
                d = evaluate(graph.metric, queries[i], graph.data[t.idx])
                _heappush!(candidates[i], NNTuple(t.idx, d, false), max_candidates)
            end
        end
    end
    knn_graph = [[pop!(candidates[i]) for _ in 1:length(candidates[i])][end:-1:end-(n_neighbors-1)]
                 for i in 1:length(candidates)]   
    # TODO: redo this to avoid repetition with `knn`
    ids = Array{Int}(undef, (n_neighbors, length(queries)))
    dists = Array{Float64}(undef, (n_neighbors, length(queries)))

    for i = 1:length(queries)
        for j in 1:n_neighbors
            ids[j, i] = knn_graph[i][j].idx
            dists[j, i] = knn_graph[i][j].dist
        end
    end
    return ids, dists
end

@inline unexpanded(heap) = sort(filter(x->!x.flag, heap.valtree))

"""
    _heappush!(heap::BinaryHeap, tup::NNTuple, max_candidates)

Try to push a neighbor `tup` to `heap`. This will fail (return `0`) if `tup` is
already in `heap`, if `tup.dist > top(heap).dist`. Otherwise return `1`.
If `length(heap) > max_candidates` after pushing, `pop` the largest candidate.
"""
function _heappush!(heap::AbstractHeap,
                    tup::NNTuple,
                    max_candidates::Int=length(heap))

    if max_candidates == 0
        @debug "max_candidates has a size of 0"
        return 0
    # case: empty heap
    elseif length(heap) == 0
        push!(heap, tup)
        return 1
    elseif length(heap) < max_candidates || tup < top(heap)
        # check if already in heap
        for i in 1:length(heap)
            exists, updated = _check_tuple(heap, i, tup)
            if updated
                return 1
            elseif exists
                return 0
            end
        end
        # push and maintain size
        _push_or_update!(heap, tup, max_candidates)
        return 1
    end
    return 0
end

@inline function _push_or_update!(h::MutableBinaryHeap, t, maxc)
    _, i = top_with_handle(h)
    h[i] = t
    nothing
end
@inline function _push_or_update!(h::BinaryHeap, t, maxc)
    push!(h, t)
    if length(h) > maxc
        pop!(h)
    end
    nothing
end

"""
Check if a tuple exists in a heap at index `i`, and optionally update its dist.
Returns (exists::Bool, updated::Bool)
"""
function _check_tuple() end

@inline function _check_tuple(h::MutableBinaryHeap, i, t)
    if h[i].idx == t.idx
        if h[i].dist == Inf
            h[i] = t
            return true, true
        end
        return true, false
    end
    return false, false
end

@inline function _check_tuple(h::BinaryHeap, i, t)
    if h.valtree[i].idx == t.idx
        return true, false
    end
    return false, false
end
