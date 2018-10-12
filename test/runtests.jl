# tests
using NNDescent
using NNDescent: _fw_neighbors, _bw_neighbors, brute_knn, _nn_descent, _update_nn
using Test
using DataStructures: mutable_binary_maxheap, top

include("knn_tree_tests.jl")
include("basic_nndescent_tests.jl")
