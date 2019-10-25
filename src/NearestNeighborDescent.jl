module NearestNeighborDescent

using DataStructures
using Distances
using LightGraphs

include("knn_graph/KNNGraphs.jl")
using .KNNGraphs
include("_descent.jl")

include("utils.jl")
include("nn_descent.jl")
include("brute_knn.jl")

export DescentGraph, ApproximateKNNGraph, HeapKNNGraph
export nndescent, search

end # module
