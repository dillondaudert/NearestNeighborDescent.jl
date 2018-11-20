module NNDescent

using Distances: Metric, Euclidean, evaluate

using DataStructures

include("heaps.jl")
include("utils.jl")
include("nn_descent.jl")
include("brute_knn.jl")

export DescentGraph, search, is_minmax_heap, minmax_heapify!

end # module
