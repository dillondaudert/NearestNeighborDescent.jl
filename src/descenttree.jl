# A simple NN tree implementation
import Random: randperm
import Base: <

struct NNTuple{R, S, T}
    idx::R
    dist::S
    flag::T
end

<(a::NNTuple, b::NNTuple) = a.dist < b.dist

struct DescentTree{V <: AbstractVector,K,R,D,M <: Metric} <: NNTree{V,M}
    data::Vector{V}
    n_neigbhors::K
    sample_rate::R
    precision::D
    metric::M
    knn_tree::Vector{MutableBinaryHeap}
end

"""
"""
function DescentTree(data::Vector{V},
                     n_neighbors::Int,
                     sample_rate::AbstractFloat,
                     precision::AbstractFloat = 0.001,
                     metric::M = Euclidean()
                     ) where {V <: AbstractArray, M <: Metric}
    n_d = length(V)
    n_p = length(data)

    knn_tree = Vector{MutableBinaryHeap{NNTuple{Int,eltype(V),Bool}, DataStructures.LessThan}}(undef, n_p)

    # build the knn_tree

    return DescentTree(data, n_neighbors, sample_rate, precision, metric, knn_tree)
end

function _nn_descent(data::Vector{V},
                     metric::Metric,
                     n_neighbors::Int,
                     sample_rate::R,
                     precision::R = 0.001,
                    ) where {V <: AbstractArray, R <: AbstractFloat}
    n_p = length(data)


    # B[v] = sample(data, k) x {<inf,true>} for all v ∈ data
    knn_tree = _init_knn_tree(data, n_neighbors)
    while true
        # NOTE: For this to be efficient, I need a data structure that has key
        #       => value pairs, where the keys are data indexes, and the values
        #       are distances.
        #       The data structure must also be ordered, sorted on values.

        # parallel for v in data
        #     old[v] = all in B[v] with false flag
        #     new[v] = rho*k items in B[v] with true flag
        #     mark sampled items in B[v] as false
        # old' = reverse(old), new' = reverse(new)
        c = 0
        # parallel for v in data
        #    old[v] = old[v] union sample(old'[v], rho*k)
        #    new[v] = newp[v] union sample(new'[v], rho*k)
        #    for u1, u2 in new[v], u1 < u2
        #        l = metric(u1, u2)
        #        // c and B[.] are synchronized
        #        c = c + updateNN(B[u1], <u2, l, true>)
        #        c = c + updateNN(B[u2], <u1, l, true>)
        if  c < sample_rate*n_neighbors*length(data)
            break
        end
    end
    return knn_tree
end

"""
Create an array of min-heaps B[j] for each j in the data.
Each minheap is initialized with `n_neighbors` random points from the data,
with distances set to `Inf` and local join flag `true`.
"""
function _init_knn_tree(data::Vector{V},
                        n_neighbors::Int) where {V <: AbstractArray}
    n_p = length(data)
    knn_tree = [mutable_binary_minheap(NNTuple{Int,eltype(V),Bool}) for _ in 1:n_p]
    for p in 1:n_p
        k_idxs = sample_neighbors(n_p, n_neighbors)
        for idx in k_idxs
            push!(knn_tree[p], NNTuple(idx, Inf, true))
        end
    end
    return knn_tree
end


function sample_neighbors(n_points::Int,
                          n_neighbors::Int,
                          sample_rate::R = 1.) where {R <: AbstractFloat}
    last = min(n_points, trunc(Int, sample_rate*n_neighbors))
    idxs = randperm(n_points)[1:last]
    return idxs
end
