# fallback method definitions for knn graph types

#######################
# ApproximateKNNGraph #
#######################

"""
    knn_diameter(g::ApproximateKNNGraph{V}, v::V)

Compute the diameter of the ball centered on `v` that covers
all of `v`s approximate k-nearest neighbors.
"""
function knn_diameter(g::ApproximateKNNGraph{V}, v::V) where V 
    neighbs = outneighbors(g, v)
    return 2 * maximum(weights(g)[neighbs, v])
end
