module NearestNeighborDescent

using DataStructures
using Distances
using LightGraphs

include("knn_graph/KNNGraphs.jl")
using .KNNGraphs
include("_utils.jl")
include("_descent.jl")

include("utils.jl")
include("nn_descent.jl")
include("brute_knn.jl")

# main exports
export DescentGraph
export nndescent, search
# KNNGraphs exports
export KNNGraphs
export ApproximateKNNGraph, HeapKNNGraph, HeapKNNGraphEdge
export knn_diameter, knn_matrices

end # module
