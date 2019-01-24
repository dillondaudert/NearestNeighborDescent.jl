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

# graph construction time
SUITE["graph"] = BenchmarkGroup(["graph", "construction"])
SUITE["graph"]["time"] = BenchmarkGroup()

for d in datasets, k in n_neighbors, m in metrics
    SUITE["graph"]["time"][d[1], k, string(m)] = @benchmarkable DescentGraph($(d[2]), $(k), $(m))
end

# graph recall

# queries per second 