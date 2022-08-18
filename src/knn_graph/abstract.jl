
#######################
# ApproximateKNNGraph #
#######################

# Abstract KNN Graph definitions
"""
ApproximateKNNGraph{V, U, D, M} subtypes are weighted, directed graphs where each vertex
has exactly `k` forward edges with weights of type `U`.

`D` is the type of the dataset corresponding to this graph, and `M` is a `Distances.PreMetric`
with result type matching `U`.
"""
abstract type ApproximateKNNGraph{V<:Integer,
                                  U<:Real,
                                  D<:AbstractVector,
                                  M<:PreMetric} <: AbstractGraph{V} end

# interface
Base.eltype(::ApproximateKNNGraph{V}) where V = V
# all knn graphs are directed
Graphs.is_directed(::Type{<:ApproximateKNNGraph}) = true
Graphs.is_directed(::ApproximateKNNGraph) = true

"""
    knn_diameter(g::ApproximateKNNGraph{V}, v::V)

Compute the diameter of the ball centered on `v` that covers
all of `v`s approximate k-nearest neighbors.
"""
function knn_diameter(g::ApproximateKNNGraph{V}, v::V) where V
    neighbs = outneighbors(g, v)
    return 2 * maximum(weights(g)[neighbs, v])
end
