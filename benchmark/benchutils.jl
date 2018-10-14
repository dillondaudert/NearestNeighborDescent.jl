# Utilities for benchmarking NN Descent
using BenchmarkTools
using NNDescent: _nn_descent, brute_knn

recall(nn, true_nn) = sum(_recall(nn[i,:], true_nn[i,:]) for i in 1:size(nn,1))/size(nn,1)
_recall(π, πₜ) = length(intersect(π, πₜ))/length(πₜ)

function do_benchmark()
    return
end
