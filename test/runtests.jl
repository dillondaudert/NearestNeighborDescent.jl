# tests
using NNDescent
using NNDescent: NNTuple, _neighbors, brute_knn, _update_nn!, _heappush!
using DataStructures
using Test

include("knn_tree_tests.jl")
include("basic_nndescent_tests.jl")
include("heaps_tests.jl")
