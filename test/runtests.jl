# tests
using NNDescent
using NNDescent: _fw_neighbors, _bw_neighbors, brute_knn
using Test
using DataStructures: mutable_binary_maxheap

include("knn_tree_tests.jl")
include("basic_nndescent_tests.jl")
