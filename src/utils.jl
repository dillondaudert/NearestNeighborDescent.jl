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
