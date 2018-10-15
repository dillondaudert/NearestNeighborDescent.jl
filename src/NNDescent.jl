module NNDescent

using NearestNeighbors
import NearestNeighbors.NNTree

using Distances
import Distances: Metric, evaluate

include("utils.jl")
include("descent_tree.jl")
include("brute_knn.jl")

export DescentTree

end # module
