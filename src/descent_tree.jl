# A simple NN tree implementation

struct DescentTree{V <: AbstractVector,K,M <: Metric} <: NNTree{V,M}
    data::Vector{V}
    nneighbors::K
    metric::M
    knntree::Matrix{NNTuple{Int,eltype(V)}}
end

"""
    DescentTree(data, nneighbors [, metric = Euclidean()]) -> descenttree
"""
function DescentTree(data::Vector{V},
                     nneighbors::Int,
                     metric::Metric = Euclidean()
                    ) where {V <: AbstractVector}
    knntree = _nn_descent(data, metric, nneighbors)
    DescentTree(data, nneighbors, metric, knntree)
end

"""
Return a kNN graph for the input data according to the given metric.
"""
function _nn_descent(data::Vector{V},
                     metric::Metric,
                     k::Int,
                     precision::R = 0.001
                    ) where {V <: AbstractArray, R <: AbstractFloat}

    np = length(data)
    # initialize with random neighbors
    knn_tree = _init_knn_tree(data, k)

    # until no further updates
    while true
        # get the fw and bw neighbors of each point
        old_fw, fw, old_bw, bw = _neighbors(knn_tree)
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
                        c = c + _update_nn!(knn_tree[u₁], NNTuple(u₂, d))
                        c = c + _update_nn!(knn_tree[u₂], NNTuple(u₁, d))
                    end
                end
                for u₂ ∈ old_neighbors[i]
                    # one point is new
                    if i ≠ u₁ && i ≠ u₂
                        d = evaluate(metric, data[u₁], data[u₂])
                        c = c + _update_nn!(knn_tree[u₁], NNTuple(u₂, d))
                        c = c + _update_nn!(knn_tree[u₂], NNTuple(u₁, d))
                    end
                end
            end

        end
        if c < precision*k*np
            break
        end
    end
    knn_ids = zeros(Int, (np, k))
    for i = 1:np, j = 1:k
        knn_ids[i, j] = pop!(knn_tree[i]).idx
    end

    return knn_ids
end

"""
Update the nearest neighbors of point `v`.
"""
function _update_nn!(v_knn,
                     u::NNTuple{S, T}) where {S, T}

    exists, updated = false, false
    if u.dist < v_knn[end].dist
        # this point is closer than the furthest nearest neighbor
        # either this point exists - we update if Inf, otherwise no update
        #   or this point is not already a NN, so we add.

        # check if point in kNN and update if distance is Inf
        for i in 1:length(v_knn)
            if v_knn[i].idx == u.idx
                exists = true
                if v_knn[i].dist == Inf
                    v_knn[i] = u
                    updated = true
                    break
                end
            end
        end
        if !exists
            # u is a new nearest neighbor
            v_knn[end] = u
            updated = true
        end

        if updated
            # use insertion sort since the array is mostly sorted (only 1 change)
            sort!(v_knn, alg=InsertionSort)
            return 1
        end
    end
    return 0
end

"""
Get the neighbors of each point in a KNN tree, `knn`,
as an array of ids.
"""
function _neighbors(knn)
    old_fw_neighbors = [Vector{Int}() for _ in 1:length(knn)]
    new_fw_neighbors = [Vector{Int}() for _ in 1:length(knn)]
    old_bw_neighbors = [Vector{Int}() for _ in 1:length(knn)]
    new_bw_neighbors = [Vector{Int}() for _ in 1:length(knn)]
    for i in 1:length(knn)
        for j in 1:length(knn[i])
            # add incoming edge ith -> jth.idx
            if knn[i][j].flag
                # denote this neighbor has participated in local join
                knn[i][j].flag = false
                append!(new_fw_neighbors[i], knn[i][j].idx)
                append!(new_bw_neighbors[knn[i][j].idx], i)
            else
                append!(old_fw_neighbors[i], knn[i][j].idx)
                append!(old_bw_neighbors[knn[i][j].idx], i)
            end
        end
    end
    return old_fw_neighbors, new_fw_neighbors, old_bw_neighbors, new_bw_neighbors
end

function _init_knn_tree(data::Vector{V},
                        n_neighbors::Int) where {V <: AbstractArray}
    np = length(data)
    knn_tree = [fill(NNTuple(-1, Inf), (n_neighbors)) for _ in 1:np]
    for i in 1:np
        k_idxs = sample_neighbors(np, n_neighbors, exclude=[i])
        for j in 1:length(k_idxs)
            knn_tree[i][j] = NNTuple(k_idxs[j], Inf)
        end
    end
    return knn_tree
end