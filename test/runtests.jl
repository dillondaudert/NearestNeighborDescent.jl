# tests
using NearestNeighborDescent
using NearestNeighborDescent: NNTuple, _neighbors, brute_knn, _heappush!
using NearestNeighborDescent.KNNGraphs: HeapKNNGraph, HeapKNNGraphEdge, flag, weight, knn_diameter
using LightGraphs
using DataStructures
using Distances
using Test
using Random

include("utils_tests.jl")
include("nndescent_tests.jl")
@testset "KNN Graph tests" begin
    include("knn_graph/graph_tests.jl")
    include("knn_graph/edge_tests.jl")
    include("knn_graph/utils_tests.jl")
end
