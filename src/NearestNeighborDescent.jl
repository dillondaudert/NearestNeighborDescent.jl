module NearestNeighborDescent

using DataStructures
using Distances
using LightGraphs

include("knn_graph/KNNGraphs.jl")
using .KNNGraphs
include("utils.jl")
include("descent.jl")

export nndescent, search
# KNNGraphs exports
export KNNGraphs
export ApproximateKNNGraph, HeapKNNGraph, HeapKNNGraphEdge, LockHeapKNNGraph
export knn_diameter, knn_matrices

end # module
