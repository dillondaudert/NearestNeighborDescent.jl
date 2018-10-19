# tests
using NNDescent
using NNDescent: NNTuple, _neighbors, brute_knn, _nn_descent, _update_nn!
using DataStructures
using Test

include("knn_tree_tests.jl")
include("basic_nndescent_tests.jl")
