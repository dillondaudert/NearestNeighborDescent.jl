using NearestNeighborDescent
using BenchmarkTools
using Distances
using MLDatasets
using PkgBenchmark

include("benchutils.jl")

const SUITE = BenchmarkGroup()

# graph construction time
SUITE["graph"] = BenchmarkGroup(["graph", "construction"])

# randomly generated dataset benchmarks
rand_vecs = [2. .* rand(100) .- 1 for _ in 1:5000]
rand_mats = 2. .* rand(100, 5000) .- 1
ham_vecs = [rand([0, 1], 100) for _ in 1:5000]
ham_mats = rand([0, 1], 100, 5000)

SUITE["graph"]["random"] = BenchmarkGroup()
SUITE["graph"]["random"][:vectors, :euclidean] = @benchmarkable DescentGraph($rand_vecs, 10, Euclidean())
SUITE["graph"]["random"][:matrices, :euclidean] = @benchmarkable DescentGraph($rand_mats, 10, Euclidean())
SUITE["graph"]["random"][:vectors, :cosine] = @benchmarkable DescentGraph($rand_vecs, 10, CosineDist())
SUITE["graph"]["random"][:matrices, :cosine] = @benchmarkable DescentGraph($rand_mats, 10, CosineDist())
SUITE["graph"]["random"][:vectors, :hamming] = @benchmarkable DescentGraph($ham_vecs, 10, Hamming())
SUITE["graph"]["random"][:matrices, :hamming] = @benchmarkable DescentGraph($ham_mats, 10, Hamming())

# real-world dataset benchmarks
SUITE["graph"]["real"] = BenchmarkGroup()
FMNIST_data = FashionMNIST.convert2features(FashionMNIST.traintensor(Float64))
FMNIST_queries = FashionMNIST.convert2features(FashionMNIST.testtensor(Float64))

MNIST_data = MNIST.convert2features(MNIST.traintensor(Float64))
MNIST_queries = MNIST.convert2features(MNIST.testtensor(Float64))

SUITE["graph"]["real"][:mnist] = @benchmarkable DescentGraph($MNIST_data, 10, Euclidean())
SUITE["graph"]["real"][:fmnist] = @benchmarkable DescentGraph($FMNIST_data, 5, Euclidean())

# queries per second 
SUITE["query"] = BenchmarkGroup(["query", "search"])
rand_query_vecs = [2. .* rand(100) .- 1 for _ in 1:1000]
rand_query_mats = 2. .* rand(100, 1000) .- 1
ham_query_vecs = [rand([0, 1], 100) for _ in 1:1000]
ham_query_mats = rand([0, 1], 100, 1000)

SUITE["query"]["random"] = BenchmarkGroup()
SUITE["query"]["random"][:vectors, :euclidean] = @benchmarkable search(graph, $rand_query_vecs, 10, 10) setup=(graph=DescentGraph($rand_vecs, 10, Euclidean()))
SUITE["query"]["random"][:matrices, :euclidean] = @benchmarkable search(graph, $rand_query_mats, 10, 10) setup=(graph=DescentGraph($rand_mats, 10, Euclidean()))
SUITE["query"]["random"][:vectors, :cosine] = @benchmarkable search(graph, $rand_query_vecs, 10, 10) setup=(graph=DescentGraph($rand_vecs, 10, CosineDist()))
SUITE["query"]["random"][:matrices, :cosine] = @benchmarkable search(graph, $rand_query_mats, 10, 10) setup=(graph=DescentGraph($rand_mats, 10, CosineDist()))
SUITE["query"]["random"][:vectors, :hamming] = @benchmarkable search(graph, $ham_query_vecs, 10, 10) setup=(graph=DescentGraph($ham_vecs, 10, Hamming()))
SUITE["query"]["random"][:matrices, :hamming] = @benchmarkable search(graph, $ham_query_mats, 10, 10) setup=(graph=DescentGraph($ham_mats, 10, Hamming()))