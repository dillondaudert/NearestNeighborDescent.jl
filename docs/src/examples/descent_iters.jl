# print information over iterations of nndescent
using NearestNeighborDescent
using Graphs
using Distances
using MLDatasets
using Statistics

include("../benchmark/benchutils.jl")

function diameter_stats(graph)
    diameters = [knn_diameter(graph, v) for v in vertices(graph)]
    min_diam, med_diam, max_diam = minimum(diameters), median(diameters), maximum(diameters)
    mean_diam = mean(diameters)
    return min_diam, mean_diam, med_diam, max_diam
end

num_new_edges(graph) = sum(KNNGraphs.flag(e) for e in edges(graph))

function do_iter(graph, data, metric, sample_rate, true_indices)
    c = NearestNeighborDescent.local_join!(graph, data, metric; sample_rate=sample_rate)
    min_diam, mean_diam, med_diam, max_diam = diameter_stats(graph)
    nne = num_new_edges(graph)
    indices = knn_matrices(graph)[1]
    rec = recall(indices, true_indices)
    return c, (min_diam, mean_diam, med_diam, max_diam), nne, rec
end

function join_with_history(data, k, metric, sample_rate, true_indices)
    graph = HeapKNNGraph(data, k, metric)
    indices = knn_matrices(graph)[1]
    count_history = []
    diam_history = []
    new_edge_history = []
    recall_history = []

    count = ne(graph)
    push!(count_history, "iter_0"=>count)
    push!(diam_history, "iter_0"=>diameter_stats(graph))
    push!(new_edge_history, "iter_0"=>num_new_edges(graph))
    recall₀ = recall(indices, true_indices)
    push!(recall_history, "iter_0"=>recall₀)
    iter = 1
    while count > 0
        count, diams, nne, rec = do_iter(graph, data, metric, sample_rate, true_indices)
        push!(count_history, "iter_$iter"=>count)
        push!(diam_history, "iter_$iter"=>diams)
        push!(new_edge_history, "iter_$iter"=>nne)
        push!(recall_history, "iter_$iter"=>rec)
        iter += 1
    end
    return count_history, diam_history, new_edge_history, recall_history
end

mnist_x_arr = [col for col in
    eachcol(MNIST.convert2features(MNIST.traintensor(Float64, 1:10000)))]
k = 20
metric = SqEuclidean()

true_graph = NearestNeighborDescent.brute_knn(mnist_x_arr, metric, k)
true_indices = getindex.(true_graph, 1)

c_hist_100, d_hist_100, e_hist_100, r_hist_100 =
    join_with_history(mnist_x_arr, k, metric, 1, true_indices)

c_hist_25, d_hist_25, e_hist_25, r_hist_25 =
    join_with_history(mnist_x_arr, k, metric, .25, true_indices)
