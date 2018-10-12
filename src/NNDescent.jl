module NNDescent

using NearestNeighbors
import NearestNeighbors.NNTree

using Distances
import Distances: Metric, evaluate

using DataStructures
import DataStructures: mutable_binary_minheap

include("basic_nndescent.jl")
include("brute_knn.jl")

export NNDescentTree, _NNTuple

end # module
