"""
Brute-force (𝒪(n²)) kNN search algorithm.
Returns an KxN array of tuples (index, distance) of the k nearest neighbors 
for each point in `data`.
"""
function brute_knn(data::Vector{V},
                   metric::M,
                   k::Int) where {V <: AbstractArray, M <: Metric}

    np = length(data)
    Dtype = code_typed(evaluate, (M, V, V))[2]
    distances = Matrix{NNTuple{Int, Dtype}}(undef, np, np)

    @inbounds @fastmath for i = 1:np, j = 1:np
        d = evaluate(metric, data[i], data[j])
        distances[i, j] = NNTuple(j, d)
    end

    # start at 2 as index 1 is the distance to itself (once sorted)
    knn_graph_tuples = sort(distances, dims=2)[1:end, 2:k+1]
    knn_graph = Matrix{Tuple{Int, Dtype}}(undef, k, np)
    for i = 1:k, j = 1:np
        knn_graph[i, j] = (knn_graph_tuples[j, i].idx,
                           knn_graph_tuples[j, i].dist)
    end
    return knn_graph
end
