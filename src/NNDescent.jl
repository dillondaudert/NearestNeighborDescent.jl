module NNDescent

using NearestNeighbors
import NearestNeighbors.NNTree

using Distances
import Distances: Metric, evaluate

using DataStructures
import DataStructures: mutable_binary_minheap

include("descent_tree.jl")
include("utils.jl")
include("brute_knn.jl")

export DescentTree, NNTuple

end # module
