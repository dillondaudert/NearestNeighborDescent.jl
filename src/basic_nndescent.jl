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
    n_p = length(data)

    knn_tree = Vector{MutableBinaryHeap{_NNTuple{Int64,eltype(V)}, DataStructures.GreaterThan}}(undef, n_p)

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

    n_p = length(data)
    # initialize with random neighbors
    knn_tree = _init_knn_tree(data, k)

    # until no further updates
    while true
        # get the fw and bw neighbors of each point
        _neighbors = [union(_fw_neighbors(knn_tree)[i],
                      _bw_neighbors(knn_tree)[i]) for i in 1:n_p]
        c = 0
        # calculate distances to neighbors' neighbors
        for i in 1:n_p
            for u₁ ∈ _neighbors[i], u₂ ∈ _neighbors[u₁]
                if i < u₂
                    d = evaluate(metric, data[i], data[u₂])
                    c = c + _update_nn(knn_tree, i, _NNTuple(u₂, d))
                end
            end
        end
        if c == 0
            break
        end
    end
    return knn_tree
end

"""
Update the nearest neighbors of point `v`.
If `u` is closer than the furthest k-NN of `v`, it will be added
to the neighbors of `v`.
"""
function _update_nn(knn_tree::Vector{R},
                    v::Int,
                    u::_NNTuple{S, T}) where {R, S, T}
    n, i = top_with_handle(knn_tree[v])

    if u.dist < n.dist
        update!(knn_tree[v], i, u)
        return 1
    end
    return 0
end

"""
Get the backward neighbors of each point in a KNN tree, `knn`,
as an array of ids.
"""
function _bw_neighbors(knn::Vector{R}) where {R}
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
function _fw_neighbors(knn::Vector{R}) where {R}
    fw_neighbors = [[knn[i][j].idx for j in 1:length(knn[i])]
                        for i in 1:length(knn)]
    return fw_neighbors
end

function _init_knn_tree(data::Vector{V},
                        n_neighbors::Int) where {V <: AbstractArray}
    n_p = length(data)
    knn_tree = [mutable_binary_maxheap(_NNTuple{Int, eltype(V)}) for _ in 1:n_p]
    for p in 1:n_p
        k_idxs = sample_neighbors(n_p, n_neighbors, exclude=[p])
        for idx in k_idxs
            push!(knn_tree[p], _NNTuple(idx, Inf))
        end
    end
    return knn_tree
end

"""
Sample `n_neighbors` elements from a set of ints `1:n_points`.
The ints in `exclude` won't be sampled.
"""
function sample_neighbors(n_points::Int,
                          n_neighbors::Int,
                          sample_rate::R = 1.;
                          exclude::Vector{Int} = Vector{Int}()) where {R <: AbstractFloat}
    last = min(n_points-length(exclude), trunc(Int, sample_rate*n_neighbors))
    idxs = setdiff(randperm(n_points), exclude)[1:last]
    return idxs
end
