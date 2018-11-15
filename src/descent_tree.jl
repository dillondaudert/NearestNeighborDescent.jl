# A simple NN tree implementation

struct DescentTree{V <: AbstractVector,K,M,S <: AbstractVector} <: NNTree{V,M}
    data::Vector{V}
    nneighbors::K
    metric::M
    graph::Vector{S}
    function DescentTree(data::Vector{V},
                         nneighbors::K,
                         metric::M,
                         graph::Vector{S}) where {V <: AbstractVector,
                                                  K <: Integer,
                                                  M <: Metric,
                                                  S <: AbstractVector}
        new{V, K, M, S}(data, nneighbors, metric, graph)
    end
end

"""
    DescentTree(data, nneighbors [, metric = Euclidean()]) -> descenttree
"""
function DescentTree(data::Vector{V},
                     n_neighbors::Int,
                     metric::Metric = Euclidean()
                    ) where {V <: AbstractVector}
    graph = build_graph(data, metric, n_neighbors)
    DescentTree(data, n_neighbors, metric, graph)
end

"""
    knn(tree::DescentTree) -> ids, dists

Return the prebuilt kNN as a tuple `(ids, dists)` where `ids` is an `KxN` matrix
of integer indices and `dists` is an `KxN` matrix of distances.
"""
function knn(tree::DescentTree)
    np, k = length(tree.graph), tree.nneighbors
    ids = Array{Int}(undef, (k, np))
    dists = Array{Float64}(undef, (k, np))

    for i = 1:np
        for j in 1:k
            ids[j, i] = tree.graph[i][j].idx
            dists[j, i] = tree.graph[i][j].dist
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
                        c += _update_nn!(knn_heaps[u₁], NNTuple(u₂, d))
                        c += _update_nn!(knn_heaps[u₂], NNTuple(u₁, d))
                    end
                end
                for u₂ ∈ old_neighbors[i]
                    # one point is new
                    if i ≠ u₁ && i ≠ u₂
                        d = evaluate(metric, data[u₁], data[u₂])
                        c += _update_nn!(knn_heaps[u₁], NNTuple(u₂, d))
                        c += _update_nn!(knn_heaps[u₂], NNTuple(u₁, d))
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
Update the nearest neighbors of point `v`.
"""
function _update_nn!(v_knn,
                     u::NNTuple{S, T}) where {S, T}

    if u.dist < top(v_knn).dist
        # this point is closer than the furthest nearest neighbor
        # either this point exists - we update if Inf, otherwise no update
        #   or this point is not already a NN, so we add.

        # check if point in kNN and update if distance is Inf
        for i in 1:length(v_knn)
            if v_knn[i].idx == u.idx
                # update distance
                if v_knn[i].dist == Inf
                    v_knn[i] = u
                    return 1
                end
                # no update
                return 0
            end
        end
        # u is a new nearest neighbor
        _, i = top_with_handle(v_knn)
        v_knn[i] = u
        return 1
    end
    return 0
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
    search(tree::DescentTree, queries::Vector{V}, n_neighbors, queue_size) -> indices, distances

Search the kNN `tree` for the nearest neighbors of the points in `queries`.
`queue_size` controls how large the candidate queue should be as a multiple of
`n_neighbors`. Larger values increase accuracy at the cost of speed, default=1.
"""
function search(tree::DescentTree,
                queries::Vector{V},
                n_neighbors::Integer,
                queue_size::Real = 1.,
                ) where {V <: AbstractArray}
    max_candidates = trunc(Int, n_neighbors*queue_size)
    candidates = [binary_maxheap(NNTuple{Int, Float64}) for _ in 1:length(queries)]
    for i in eachindex(queries)
        # init
        j =  rand(1:length(tree.data))
        d = evaluate(tree.metric, queries[i], tree.data[j])
        _heappush!(candidates[i], NNTuple(j, d, false), max_candidates)

        while true
            unexp = unexpanded(candidates[i])
            if length(unexp) == 0
                break
            end
            # expand closest unexpanded neighbor
            unexp[1].flag = true
            #unexp[1].idx is idx in data of candidate neighbor to queries[i]
            # tree.graph[unexp[1].idx] is an array of NNtuples of the approx kNN
            for t in tree.graph[unexp[1].idx]
                d = evaluate(tree.metric, queries[i], tree.data[t.idx])
                _heappush!(candidates[i], NNTuple(t.idx, d, false), max_candidates)
            end

        end
    end
    return candidates
end

@inline unexpanded(heap) = sort(filter(x->!x.flag, heap.valtree))

"""
    _heappush!(heap::BinaryHeap, tup::NNTuple, max_candidates)

Try to push a neighbor `tup` to `heap`. This will fail (return `0`) if `tup` is
already in `heap`, if `tup.dist > top(heap).dist`. Otherwise return `1`.
If `length(heap) > max_candidates` after pushing, `pop` the largest candidate.
"""
function _heappush!(heap::BinaryHeap{NNTuple{S, T}},
                    tup::NNTuple{S, T},
                    max_candidates::Int) where {S, T}

    if max_candidates == 0
        @debug "max_candidates has a size of 0"
        return 0
    # case: empty heap
    elseif length(heap) == 0
        push!(heap, tup)
        return 1
    elseif length(heap) < max_candidates || tup < top(heap)
        # check if already in heap
        for i in eachindex(heap.valtree)
            if heap.valtree[i].idx == tup.idx
                return 0
            end
        end
        # push and maintain size
        push!(heap, tup)
        if length(heap) > max_candidates
            pop!(heap)
        end
        return 1
    end
    return 0
end
