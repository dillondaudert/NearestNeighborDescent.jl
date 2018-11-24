import Random: randperm
import Base: <, isless

mutable struct NNTuple{R, S}
    idx::R
    dist::S
    flag::Bool
end
NNTuple(a, b) = NNTuple(a, b, true)

<(a::NNTuple, b::NNTuple) = a.dist < b.dist
isless(a::NNTuple, b::NNTuple) = <(a, b)

"""
Sample `n_neighbors` elements from a set of ints `1:npoints`.
The ints in `exclude` won't be sampled.
"""
function sample_neighbors(npoints::Int,
                          n_neighbors::Int,
                          sample_rate::R = 1.;
                          exclude::Vector{Int} = Vector{Int}()) where {R <: AbstractFloat}
    # num neighbors to sample = max(ρK, N)
    last = min(npoints-length(exclude), trunc(Int, sample_rate*n_neighbors))
    idx_gen = Iterators.take((i for i ∈ randperm(npoints) if i ∉ exclude), last)
    return collect(idx_gen)
end

function make_knn_heaps(data::Vector{V},
                        n_neighbors::Int,
                        metric::M) where {V <: AbstractArray,
                                          M <: SemiMetric}
    np = length(data)
    D = result_type(metric, data[1], data[1])
    knn_heaps = [mutable_binary_maxheap(NNTuple{Int, D}) for _ in 1:np]
    for i in 1:np
        k_idxs = sample_neighbors(np, n_neighbors, exclude=[i])
        for j in k_idxs
            d = evaluate(metric, data[i], data[j])
            _heappush!(knn_heaps[i], NNTuple(j, d), n_neighbors)
            _heappush!(knn_heaps[j], NNTuple(i, d), n_neighbors)
        end
    end
    return knn_heaps
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

function min_flagged(heap)
    min_tup = NNTuple(-1, typemax(typeof(heap.valtree[1].dist)))
    for t in heap.valtree
        if t < min_tup && !t.flag
            min_tup = t
        end
    end
    return min_tup
end

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
Push `tup` onto `heap` without checkout if it already exists in the heap.
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
