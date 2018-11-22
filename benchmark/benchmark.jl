using NNDescent
using NNDescent: brute_knn, brute_search, NNTuple
using PkgBenchmark

include("benchutils.jl")

const SUITE = BenchmarkGroup()

SUITE["nndescent"] = BenchmarkGroup("constructor", "search")
SUITE["nndescent"]["fmnist"] = @benchmarkable DescentGraph($FMNIST_data, 10)
SUITE["nndescent"]["mnist"] = @benchmarkable DescentGraph($MNIST_data, 10)
SUITE["brute"]["fmnist"] = @benchmarkable brute_knn($FMNIST_data, 10)
SUITE["brute"]["mnist"] = @benchmarkable brute_knn($MNIST_data, 10)