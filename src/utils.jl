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
