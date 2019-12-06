using NearestNeighbors
using NearestNeighborDescent
using Distances
using MLDatasets
using Statistics

recall(true_ids, cand_ids) = mean(_recall(trues, cands) for (trues, cands) in zip(true_ids, cand_ids))
_recall(true_ids, cand_ids) = length(intersect(true_ids, cand_ids))/length(true_ids)

mnist_x = MNIST.convert2features(MNIST.traintensor(Float64))
fmnist_x = FashionMNIST.convert2features(FashionMNIST.traintensor(Float64))

mnist_y = MNIST.convert2features(MNIST.testtensor(Float64))
fmnist_y = FashionMNIST.convert2features(FashionMNIST.testtensor(Float64))

mnist_tree = KDTree(mnist_y; leafsize=50)
fmnist_tree = KDTree(fmnist_y; leafsize=50)

mnist_graph = nndescent(mnist_y, 20, Euclidean())
fmnist_graph = nndescent(fmnist_y, 20, Euclidean())

recalls = []

for (name, graph, tree, data) in [(:mnist, mnist_graph, mnist_tree, mnist_y), (:fmnist, fmnist_graph, fmnist_tree, fmnist_y)]

    nn_ids, _ = knn(tree, data, 21, true)
    true_ids = [v[2:end] for v in nn_ids]

    cand_nn_ids, _ = knn_matrices(graph)
    cand_ids = collect(eachcol(cand_nn_ids))

    push!(recalls, name=>recall(true_ids, cand_ids))
end

recalls
