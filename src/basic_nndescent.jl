# A simple NN tree implementation
import Random: randperm
import Base: <, isless

struct _NNTuple{R, S}
    idx::R
    dist::S
end

<(a::_NNTuple, b::_NNTuple) = a.dist < b.dist
isless(a::_NNTuple, b::_NNTuple) = <(a, b)

struct NNDescentTree{V <: AbstractVector,K,M <: Metric} <: NNTree{V,M}
    data::Vector{V}
    n_neigbhors::K
    metric::M
    knn_tree::Vector{MutableBinaryHeap}
end

"""
"""
function NNDescentTree(data::Vector{V},
                       n_neighbors::Int,
                       metric::M = Euclidean()
                      ) where {V <: AbstractArray, M <: Metric}
    n_d = length(V)
    np = length(data)

    knn_tree = Vector{MutableBinaryHeap{_NNTuple{Int64,eltype(V)}, DataStructures.GreaterThan}}(undef, np)

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
                if i < u₂
                    d = evaluate(metric, data[i], data[u₂])
                    c = c + _update_nn!(knn_tree, i, _NNTuple(u₂, d),
                                       u₂ in fw[i])
                    c = c + _update_nn!(knn_tree, u₂, _NNTuple(i, d),
                                       i in fw[u₂])
                end
            end
        end
        if c == 0
            break
        end
    end
    @show knn_tree
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
                     u::_NNTuple{S, T},
                     exists::Bool = false) where {S, T}

    if exists
        # update value of u in v_knn
        for i in 1:length(v_knn)
            if v_knn[i].idx == u.idx
                if u.dist < v_knn[i].dist
                    v_knn[i] = u
                    break
                else
                    # no update made since distances were equal
                    return 0
                end
            end
        end
    elseif u.dist < v_knn[end].dist
        # u is a new nearest neighbor
        v_knn[end] = u
    else
        # no updates, so return 0
        return 0
    end
    # use insertion sort since the array is mostly sorted (only 1 change)
    sort!(v_knn, alg=InsertionSort)
    return 1
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
    knn_tree = [fill(_NNTuple(-1, Inf), (n_neighbors)) for _ in 1:np]
    for i in 1:np
        k_idxs = sample_neighbors(np, n_neighbors, exclude=[i])
        for j in 1:length(k_idxs)
            knn_tree[i][j] = _NNTuple(k_idxs[j], Inf)
        end
    end
    return knn_tree
end

"""
Sample `n_neighbors` elements from a set of ints `1:npoints`.
The ints in `exclude` won't be sampled.
"""
function sample_neighbors(npoints::Int,
                          n_neighbors::Int,
                          sample_rate::R = 1.;
                          exclude::Vector{Int} = Vector{Int}()) where {R <: AbstractFloat}
    last = min(npoints-length(exclude), trunc(Int, sample_rate*n_neighbors))
    idxs = setdiff(randperm(npoints), exclude)[1:last]
    return idxs
end
