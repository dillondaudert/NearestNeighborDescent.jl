# tests
using NearestNeighborDescent
using NearestNeighborDescent.KNNGraphs
using Graphs
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

# NOTE: JET-based type-stability checks live in jet_tests.jl and run as a dedicated CI
# step (see .github/workflows/CI.yml), not here. JET is tightly coupled to the Julia
# compiler (the version we target requires Julia ≥ 1.12) and Pkg.test()'s sandbox does
# not expose Pkg, so they are kept out of the standard suite. Run them locally with:
#   julia --project=. test/jet_tests.jl
