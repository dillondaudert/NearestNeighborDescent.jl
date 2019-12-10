# # NNDescent on MNIST digits
# In this example, we build an approximate KNN graph over the MNIST digits and
# then use it as an index to find approximate nearest neighbors of new points.

using NearestNeighborDescent
using Distances
using BenchmarkTools
using MLDatasets
mnist_x = collect(eachcol(MNIST.convert2features(MNIST.traintensor(Float64, 1:10000))))
mnist_y = collect(eachcol(MNIST.convert2features(MNIST.testtensor(Float64, 1:1000))))

## Construct the KNN Graph
graph = nndescent(mnist_x, 20, Euclidean())

## Find neighbors of new points
inds, dists = search(graph, mnist_x, mnist_y, 20, Euclidean())
