# tests
using NearestNeighborDescent
using NearestNeighborDescent.KNNGraphs
using LightGraphs
using DataStructures
using Distances
using Test
using Random

#include("utils_tests.jl")
#include("nndescent_tests.jl")
@testset "KNN Graph tests" begin
    include("knn_graph/heap_graph_tests.jl")
    include("knn_graph/edge_tests.jl")
    include("knn_graph/utils_tests.jl")
end
