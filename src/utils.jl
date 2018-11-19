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
                        ::Type{D}=Float64) where {V <: AbstractArray}
    np = length(data)
    knn_heaps = [mutable_binary_maxheap(NNTuple{Int, D}) for _ in 1:np]
    for i in 1:np
        k_idxs = sample_neighbors(np, n_neighbors, exclude=[i])
        for j in 1:length(k_idxs)
            push!(knn_heaps[i], NNTuple(k_idxs[j], convert(D, Inf)))
        end
    end
    return knn_heaps
end
