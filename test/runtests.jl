# tests
using NNDescent
using NNDescent: NNTuple, _neighbors, brute_knn, _heappush!
using DataStructures
using Test

include("utils_tests.jl")
include("nndescent_tests.jl")
include("heaps_tests.jl")
