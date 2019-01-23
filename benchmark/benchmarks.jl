using NearestNeighborDescent
using NearestNeighborDescent: brute_knn, brute_search
using BenchmarkTools
using Distances
using PkgBenchmark

include("benchutils.jl")

const SUITE = BenchmarkGroup()

datasets = ("rand"=>rand_data, 
            "mnist"=>MNIST_data,
            "fmnist"=>FMNIST_data,) 
n_neighbors = (5, 10)
metrics = (Euclidean(),)

SUITE["graph"] = BenchmarkGroup(["graph", "construction"])
SUITE["graph"]["time"] = BenchmarkGroup()

for d in datasets, k in n_neighbors, m in metrics
    SUITE["graph"]["time"][d[1], k, string(m)] = @benchmarkable DescentGraph($(d[2]), $(k), $(m))
end

#SUITE["search"] = BenchmarkGroup(["search", "query"])
#SUITE["search"]["..."]
#SUITE["brute"] = BenchmarkGroup(["constructor"])
#SUITE["brute"]["rand"] = @benchmarkable brute_knn($rand_data, Euclidean(), 10)
#SUITE["brute"]["fmnist"] = @benchmarkable brute_knn($FMNIST_data, Euclidean(), 10)
#SUITE["brute"]["mnist"] = @benchmarkable brute_knn($MNIST_data, Euclidean(), 10)