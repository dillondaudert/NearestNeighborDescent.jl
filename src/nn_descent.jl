# A simple NN graph implementation

struct DescentGraph{V <: AbstractVector,M,S <: Tuple}
    data::Vector{V}
    metric::M
    graph::Matrix{S}
end

"""
    DescentGraph(data, n_neighbors [, metric = Euclidean()])

Build an approximate kNN graph of `data` using nearest neighbor descent. 
"""
function DescentGraph(data::Vector{V},
                      n_neighbors::Integer,
                      metric::SemiMetric = Euclidean(),
                      max_iters::Integer = 10,
                      sample_rate::R = 1.,
                      precision::R = .001
                     ) where {V <: AbstractVector, R <: AbstractFloat}
    DescentGraph(data, 
                 metric, 
                 build_graph(data, n_neighbors, metric, max_iters, sample_rate, precision))
end

"""
Return a kNN graph for the input data according to the given metric.
"""
function build_graph(data::Vector{V},
                     k::Integer,
                     metric::M,
                     max_iters::Integer,
                     sample_rate::R,
                     precision::R
                    ) where {V <: AbstractArray, 
                             M <: SemiMetric, 
                             R <: AbstractFloat}

    np = length(data)
    # initialize with random neighbors
    Dtype = result_type(metric, data[1], data[1])
    knn_heaps = make_knn_heaps(data, k, metric)

    # until no further updates
    for i = 1:max_iters
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
        if c <= precision*k*np
            break
        end
    end
    knn_graph = Matrix{Tuple{Int, Dtype}}(undef, k, np)
    for j in 1:np
        rev_nns = [pop!(knn_heaps[j]) for _ in 1:length(knn_heaps[j])]
        for i in 1:k
            knn_graph[i, j] = (rev_nns[end-(i-1)].idx, rev_nns[end-(i-1)].dist)
        end
    end
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
    Dtype = result_type(graph.metric, queries[1], queries[1])
    candidates = [binary_maxheap(NNTuple{Int, Dtype}) for _ in 1:length(queries)]
    for i in eachindex(queries)
        # init
        seen = fill(false, length(graph.data))
        j =  rand(1:length(graph.data))
        d = evaluate(graph.metric, queries[i], graph.data[j])
        _heappush!(candidates[i], NNTuple(j, d, false), max_candidates)
        seen[j] = true
        
        while true
            unexp = min_flagged(candidates[i])
            if unexp.idx == -1
                break
            end
            # expand closest unexpanded neighbor
            unexp.flag = true
            for t in graph.graph[:,unexp.idx]
                if !seen[t[1]]
                    seen[t[1]] = true
                    d = evaluate(graph.metric, 
                                 queries[i], 
                                 graph.data[t[1]])
                    _unchecked_heappush!(candidates[i], 
                                         NNTuple(t[1], d, false),
                                         max_candidates)
                end
            end
        end
    end
    knn_graph = [[pop!(candidates[i]) for _ in 1:length(candidates[i])][end:-1:end-(n_neighbors-1)]
                 for i in 1:length(candidates)]   
    ids = Array{Int}(undef, (n_neighbors, length(queries)))
    dists = Array{Dtype}(undef, (n_neighbors, length(queries)))

    for i = 1:length(queries)
        for j in 1:n_neighbors
            ids[j, i] = knn_graph[i][j].idx
            dists[j, i] = knn_graph[i][j].dist
        end
    end
    return ids, dists
end

function min_flagged(heap)
    min_tup = NNTuple(-1, typemax(typeof(heap.valtree[1].dist)))
    for t in heap.valtree
        if t < min_tup && !t.flag
            min_tup = t
        end
    end
    return min_tup
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
        if length(heap) == max_candidates
            _push_or_update!(heap, tup, max_candidates)
        else
            push!(heap, tup)
        end
        return 1
    end
    return 0
end

"""
Push `tup` onto `heap` without checkout if it already exists in the
heap.
"""
function _unchecked_heappush!(heap::AbstractHeap,
                              tup::NNTuple,
                              max_candidates::Int=length(heap))
    if max_candidates == 0
        @debug "max_candidates has a size of 0"
        return 0
    end
    # case: heap not full
    if length(heap) < max_candidates
        push!(heap, tup)
        return 1
    # heap full but this neighbor closer
    elseif tup < top(heap)
        # push and maintain size
        _push_or_update!(heap, tup, max_candidates)
        return 1
    end
    return 0
                        
end

@inline function _push_or_update!(h::MutableBinaryHeap, t, maxc)
    _, i = top_with_handle(h)
    h[i] = t
    return
end
@inline function _push_or_update!(h::BinaryHeap, t, maxc)
    push!(h, t)
    if length(h) > maxc
        pop!(h)
    end
    return
end

"""
Check if a tuple exists in a heap at index `i`, and optionally update its dist.
Returns (exists::Bool, updated::Bool)
"""
function _check_tuple() end

@inline function _check_tuple(h::MutableBinaryHeap, i, t)
    if h[i].idx == t.idx
        if h[i].dist == typemax(typeof(h[i].dist))
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
