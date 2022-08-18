using Pkg
tempdir = mktempdir()
Pkg.activate(tempdir)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.add(["BenchmarkTools", "PkgBenchmark", "Random", "Distances", "Graphs"])
Pkg.resolve()

using NearestNeighborDescent
using BenchmarkTools

const SUITE = BenchmarkGroup()

SUITE["knn graphs"] = include("bench_knn_graphs.jl")
SUITE["descent"] = include("bench_descent.jl")
