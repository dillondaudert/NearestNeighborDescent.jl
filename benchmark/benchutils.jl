# Utilities for benchmarking NN Descent
using BenchmarkTools
using Distances: Euclidean
using NNDescent: _nn_descent, brute_knn

recall(nn, true_nn) = sum(_recall(nn[i,:], true_nn[i,:]) for i in 1:size(nn,1))/size(nn,1)
_recall(π, πₜ) = length(intersect(π, πₜ))/length(πₜ)

function run_benchmarks()
    nn_descent_recall = []
    nn_descent_time = []
    brute_knn_time = []
    for n in [500, 1000, 2000, 4000, 8000]
        print("Benchmarking n = $n\n")
        k = 20
        data = [rand(20) for _ in 1:n]
        append!(brute_knn_time, @belapsed brute_knn($data, $Euclidean(), $k))
        append!(nn_descent_time, @belapsed _nn_descent($data, $Euclidean(), $k))
        true_nn = brute_knn(data, Euclidean(), k)
        desc_nn = _nn_descent(data, Euclidean(), k)
        append!(nn_descent_recall, recall(desc_nn, true_nn))
    end
    @show nn_descent_recall
    @show nn_descent_time
    @show brute_knn_time
end
