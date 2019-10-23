# fallback method definitions for abstract knn graph types

#######################
# ApproximateKNNGraph #
#######################

"""
    knn_diameter(g::ApproximateKNNGraph{V}, v::V; dir=:out)

Compute the diameter of the ball centered on `v` that covers
all of `v`s neighbors. `dir` is used to specify which neighbors
to consider, one of `:in`, `:out`, `:both`.
"""
function knn_diameter(g::ApproximateKNNGraph{V}, v::V; dir=:out) where V 
    if dir == :in
        neighbs = inneighbors(g, v)
        diam = max(weights(g)[v, neighbs])
    elseif dir == :out
        neighbs = outneighbors(g, v)
        diam = max(weights(g)[neighbs, v])
    elseif dir == :both
        error("Not implemented")
    else
        throw(ArgumentError("dir=$dir is not a valid value."))
    end

    return diam
    
end
