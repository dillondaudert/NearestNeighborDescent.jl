module NNDescent

using NearestNeighbors
import NearestNeighbors.NNTree

using Distances
import Distances: Metric, evaluate

using DataStructures
import DataStructures: mutable_binary_minheap

include("descenttree.jl")

export DescentTree, NNTuple
export knn

end # module
