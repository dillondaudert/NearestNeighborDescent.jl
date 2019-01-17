# some basic profiling

using Profile
using ProfileView
using NearestNeighborDescent
using MLDatasets

mnist_x = MNIST.convert2features(MNIST.traintensor(Float64))
DescentGraph(rand(5, 100), 10) # precompile
Profile.clear_malloc_data()
DescentGraph(mnist_x, 10)

