module NearestNeighborDescent

using Distances: SemiMetric, Euclidean, evaluate, result_type
using DataStructures

include("utils.jl")
include("nn_descent.jl")
include("brute_knn.jl")

export DescentGraph, search

end # module
