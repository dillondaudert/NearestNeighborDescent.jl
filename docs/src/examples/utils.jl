# Utilities for benchmarking NN Descent

recall(nn, true_nn) = sum(_recall(nn[:,i], true_nn[:,i]) for i in 1:size(nn,2))/size(nn,2)
_recall(π, πₜ) = length(intersect(π, πₜ))/length(πₜ)
