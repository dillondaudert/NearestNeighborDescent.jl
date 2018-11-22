using NNDescent
using NNDescent: brute_knn, brute_search, NNTuple
using BenchmarkTools
using Distances
using PkgBenchmark

include("benchutils.jl")

const SUITE = BenchmarkGroup()

SUITE["nndescent"] = BenchmarkGroup(["constructor"])
SUITE["nndescent"]["rand"] = @benchmarkable DescentGraph($rand_data, 10)
SUITE["nndescent"]["fmnist"] = @benchmarkable DescentGraph($FMNIST_data, 10)
SUITE["nndescent"]["mnist"] = @benchmarkable DescentGraph($MNIST_data, 10)
SUITE["brute"] = BenchmarkGroup(["constructor"])
SUITE["brute"]["rand"] = @benchmarkable brute_knn($rand_data, Euclidean(), 10)
SUITE["brute"]["fmnist"] = @benchmarkable brute_knn($FMNIST_data, Euclidean(), 10)
SUITE["brute"]["mnist"] = @benchmarkable brute_knn($MNIST_data, Euclidean(), 10)