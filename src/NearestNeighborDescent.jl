module NearestNeighborDescent

using Distances: SemiMetric, Euclidean, evaluate, result_type
using DataStructures: AbstractHeap, BinaryMaxHeap, BinaryHeap, top, pop!

include("knn_graph/KNNGraphs.jl")
using .KNNGraphs
include("_descent.jl")

include("utils.jl")
include("nn_descent.jl")
include("brute_knn.jl")

export DescentGraph, search

end # module
