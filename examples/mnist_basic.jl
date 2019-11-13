using NearestNeighborDescent
using Distances
using BenchmarkTools
using MLDatasets

mnist_x = [col for col in eachcol(MNIST.convert2features(MNIST.traintensor(Float64, 1:10000)))]
metric = SqEuclidean()

print("DescentGraph: N=10k, K=20, SqEuclidean ")
@btime DescentGraph($(mnist_x), 20, $(metric))

print("nndescent: N=10k, K=20, SqEuclidean ")
@btime nndescent($(mnist_x), 20, $(metric))
