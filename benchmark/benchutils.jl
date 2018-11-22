# Utilities for benchmarking NN Descent
using MLDatasets

recall(nn, true_nn) = sum(_recall(nn[:,i], true_nn[:,i]) for i in 1:size(nn,2))/size(nn,2)
_recall(π, πₜ) = length(intersect(π, πₜ))/length(πₜ)

rand_data = [rand(100) for _ in 1:5000]
rand_queries = [rand(100) for _ in 1:500]

reshape_mnist(data) = reshape(data, size(data)[1]*size(data)[2], size(data)[3])

FMNIST_train, _ = FashionMNIST.traindata()
FMNIST_test, _ = FashionMNIST.testdata()
FMNIST_train = reshape_mnist(FMNIST_train)
FMNIST_test = reshape_mnist(FMNIST_test)
FMNIST_data = [convert.(Float64, FMNIST_train[:,i]) for i = 1:5000]
FMNIST_queries = [convert.(Float64, FMNIST_test[:,i]) for i = 1:500]

MNIST_train, _ = MNIST.traindata()
MNIST_train = reshape_mnist(MNIST_train)
MNIST_data = [convert.(Float64, MNIST_train[:,i]) for i = 1:5000]
MNIST_test, _ = MNIST.testdata()
MNIST_test = reshape_mnist(MNIST_test)
MNIST_queries = [convert.(Float64, MNIST_test[:,i]) for i = 1:500]