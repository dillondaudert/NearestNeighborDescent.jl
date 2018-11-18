module NNDescent

using Distances: Metric, evaluate

using DataStructures

include("heaps.jl")
include("utils.jl")
include("nn_descent.jl")
include("brute_knn.jl")

export DescentGraph, knn, search, is_minmax_heap, minmax_heapify!

end # module
