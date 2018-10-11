# A simple NN tree implementation
import Random: randperm
import Base: <

struct _NNTuple{R, S}
    idx::R
    dist::S
end

<(a::_NNTuple, b::_NNTuple) = a.dist < b.dist

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

    # B[v] = sample(V,K) x {∞}, ∀ v ∈ V
    _knn = _init_knn_tree(data, k)

    while true
        _neigbhors = get_reverse(_knn)

        # B̄ = B ∪ R
        # get the set of all general neighbors of each point
        # in this case, we just add the forward neighbors to _rev_knn
        for i in 1:n_p
            for j in 1:k
                # add neighbor j.idx to neighbors of i
                union!(_neighbors[i], _knn[i][j].idx)
            end
        end
        c = 0
        # for v ∈ V
        for i in 1:n_p
            for u₁ ∈ _neighbors[i], u₂ ∈ _neighbors[u₁]
                # l = evaluate(metric, v, u₂)
                # c = c + UpdateNN(B[v], <u₂, l>)
            end
        end
        if c == 0
            break
        end
    end
    return knn_tree
end

function get_reverse(knn::R) where {R}
    reverse_neighbors = [Set([]) for _ in 1:length(knn)]
    for i in 1:length(knn)
        for j in 1:length(knn[i])
            # add incoming edge i of jth NN of i with index idx
            union!(reverse_neighbors[knn[i][j].idx], i)
        end
    end
    return reverse_neighbors
end

function _init_knn_tree(data::Vector{V},
                        n_neighbors::Int) where {V <: AbstractArray}
    n_p = length(data)
    knn_tree = [mutable_binary_maxheap(_NNTuple{Int, eltype(V)}) for _ in 1:n_p]
    for p in 1:n_p
        k_idxs = sample_neighbors(n_p, n_neighbors)
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
