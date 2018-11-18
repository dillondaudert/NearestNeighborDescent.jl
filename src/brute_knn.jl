"""
Brute-force (𝒪(n²)) kNN search algorithm.
Returns an KxN array of tuples (index, distance) of the k nearest neighbors 
for each point in `data`.
"""
function brute_knn(data::Vector{V},
                   metric::Metric,
                   k::Int) where {V <: AbstractArray}

    np = length(data)
    distances = fill(NNTuple(0, 0.), (np, np))

    @inbounds @fastmath for i = 1:np, j = 1:np
        d = evaluate(metric, data[i], data[j])
        distances[i, j] = NNTuple(j, d)
    end

    # start at 2 as index 1 is the distance to itself (once sorted)
    knn_graph_tuples = transpose(sort(distances, dims=2)[1:end,2:k+1])
    knn_graph = Matrix{Tuple{Int, Float64}}(undef, k, np)
    for i = 1:k, j = 1:np
        knn_graph[i, j] = (knn_graph_tuples[i, j].idx,
                           knn_graph_tuples[i, j].dist)
    end
    return knn_graph
end
