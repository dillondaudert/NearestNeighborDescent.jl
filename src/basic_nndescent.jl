# A simple NN tree implementation
import Random: randperm
import Base: <, isless

struct NNTuple{R, S}
    idx::R
    dist::S
end

<(a::NNTuple, b::NNTuple) = a.dist < b.dist
isless(a::NNTuple, b::NNTuple) = <(a, b)

struct DescentTree{V <: AbstractVector,K,M <: Metric} <: NNTree{V,M}
    data::Vector{V}
    n_neighbors::K
    metric::M
    knn_tree::Vector{MutableBinaryHeap}
end

"""
    DescentTree(data [,])
"""
function DescentTree(data::Vector{V},
                     n_neighbors::Int,
                     metric::Metric = Euclidean()
                    ) where {V <: AbstractArray}
    n_d = length(V)
    np = length(data)

    knn_tree = Vector{MutableBinaryHeap{NNTuple{Int64,eltype(V)}, DataStructures.GreaterThan}}(undef, np)

    # build the knn_tree

    return NNDescentTree(data, n_neighbors, sample_rate, precision, metric, knn_tree)
end

"""
Return a kNN graph for the input data according to the given metric.
"""
function _nn_descent(data::Vector{V},
                     metric::Metric,
                     k::Int,
                    ) where {V <: AbstractArray}

    np = length(data)
    # initialize with random neighbors
    knn_tree = _init_knn_tree(data, k)

    # until no further updates
    while true
        # get the fw and bw neighbors of each point
        fw = _fw_neighbors(knn_tree)
        bw = _bw_neighbors(knn_tree)
        _neighbors = [union(fw[i], bw[i]) for i in 1:np]
        c = 0
        # calculate distances to neighbors' neighbors
        for i in 1:np
            for u₁ ∈ _neighbors[i], u₂ ∈ _neighbors[u₁]
                if i ≠ u₂
                    d = evaluate(metric, data[i], data[u₂])
                    #print(u₂, " in ", fw[i], " = ", u₂ in fw[i], "\n")
                    c = c + _update_nn!(knn_tree[i], NNTuple(u₂, d))
                end
            end
        end
        if c == 0
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
Get the backward neighbors of each point in a KNN tree, `knn`,
as an array of ids.
"""
function _bw_neighbors(knn)
    bw_neighbors = [Vector{Int}() for _ in 1:length(knn)]
    for i in 1:length(knn)
        for j in 1:length(knn[i])
            # add incoming edge i of jth NN of i with index idx
            append!(bw_neighbors[knn[i][j].idx], i)
        end
    end
    return bw_neighbors
end

"""
Get the forward neighbors of each point in a KNN tree, `knn`,
as an array of ids.
"""
function _fw_neighbors(knn)
    fw_neighbors = [[knn[i][j].idx for j in 1:length(knn[i])]
                        for i in 1:length(knn)]
    return fw_neighbors
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
