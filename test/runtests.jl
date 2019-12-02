# tests
using NearestNeighborDescent
using NearestNeighborDescent.KNNGraphs
using LightGraphs
using DataStructures
using Distances
using Test
using Random
using Statistics

@testset "KNN Graph tests" begin
    include("knn_graph/heap_graph_tests.jl")
    include("knn_graph/edge_tests.jl")
    include("knn_graph/utils_tests.jl")
end
include("utils_tests.jl")
include("descent_tests.jl")
include("search_tests.jl")
