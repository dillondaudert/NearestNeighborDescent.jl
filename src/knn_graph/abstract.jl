
#######################
# ApproximateKNNGraph #
#######################

# Abstract KNN Graph definitions
"""
ApproximateKNNGraph{V, K} subtypes are weighted, directed graphs where each vertex
has exactly `k` forward edges.
"""
abstract type ApproximateKNNGraph{V, K, U <: Real} <: AbstractGraph{V} end

# interface
Base.eltype(::ApproximateKNNGraph{V, K, U}) where {V, K, U} = V
# all knn graphs are directed
LightGraphs.is_directed(::Type{<:ApproximateKNNGraph}) = true
LightGraphs.is_directed(::ApproximateKNNGraph) = true

"""
    knn_diameter(g::ApproximateKNNGraph{V}, v::V)

Compute the diameter of the ball centered on `v` that covers
all of `v`s approximate k-nearest neighbors.
"""
function knn_diameter(g::ApproximateKNNGraph{V}, v::V) where V
    neighbs = outneighbors(g, v)
    return 2 * maximum(weights(g)[neighbs, v])
end
