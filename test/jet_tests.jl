# Type-stability / correctness regression checks powered by JET.
#
# These guard against two classes of regression that the behavioural tests miss:
#   * method errors on cold branches (e.g. comparing a scalar to a heap tuple), which
#     `@report_call` surfaces without needing the branch to execute, and
#   * type instabilities / runtime dispatch in hot loops (e.g. boxed captured variables
#     in threaded code), which `@report_opt` surfaces.
#
# JET is tightly coupled to the Julia compiler and the version we target requires Julia
# ≥ 1.12, so these run as a standalone CI step (not part of runtests.jl) on the latest
# Julia only. Run locally on Julia ≥ 1.12 with:
#   julia --project=. test/jet_tests.jl

using NearestNeighborDescent
using NearestNeighborDescent.KNNGraphs
using Distances
using Random
using Test
using JET

@testset "JET type stability + correctness" begin
    Random.seed!(0)
    data = [rand(5) for _ in 1:20]
    queries = [rand(5) for _ in 1:5]
    k = 4

    # SqEuclidean is a SemiMetric (symmetric branch); Euclidean exercises the
    # asymmetric reverse-distance branch in construction and local joins.
    @testset "nndescent ($G, $(typeof(metric)))" for G in (HeapKNNGraph, LockHeapKNNGraph),
                                                       metric in (SqEuclidean(), Euclidean())
        @test isempty(JET.get_reports(@report_call nndescent(G, data, k, metric; max_iters=1)))
        @test isempty(JET.get_reports(@report_opt nndescent(G, data, k, metric; max_iters=1)))
    end

    @testset "search ($G)" for G in (HeapKNNGraph, LockHeapKNNGraph)
        graph = nndescent(G, data, k, Euclidean(); max_iters=1)
        @test isempty(JET.get_reports(@report_call search(graph, queries, 3)))
        # max_candidates < nv exercises the unseen-neighbour path that hid the
        # `isless(::Float64, ::Tuple)` bug.
        @test isempty(JET.get_reports(@report_opt search(graph, queries, 3; max_candidates=5)))
    end
end
