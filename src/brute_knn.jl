"""
Brute-force (𝒪(n²)) kNN search algorithm.
Returns an KxN array of tuples (index, distance) of the k nearest neighbors 
for each point in `data`.
"""
function brute_knn(data::Vector{V},
                   metric::M,
                   k::Int) where {V <: AbstractArray, M <: Metric}

    np = length(data)
    Dtype = code_typed(evaluate, (M, V, V))[1][2]
    distances = Matrix{NNTuple{Int, Dtype}}(undef, np, np)

    @inbounds @fastmath for j = 1:np, i = 1:np
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

"""
    brute_search(data, queries, k, metric) -> idxs, dists

Search for the nearest `k` neighbors of `q` in `queries` in `data`.
"""
function brute_search(data::Vector{V}, 
                      queries::Vector{V}, 
                      k::Integer, 
                      metric::M=Euclidean()) where {V <: AbstractVector,
                                                    M <: Metric}
    np = length(data)
    nq = length(queries)
    Dtype = code_typed(evaluate, (M, V, V))[1][2]
    distances = Matrix{NNTuple{Int, Dtype}}(undef, np, nq)
    
    @inbounds @fastmath for i = 1:nq, j = 1:np
        d = evaluate(metric, queries[i], data[j])
        distances[j, i] = NNTuple(j, d)
    end
    
    knn_graph_tuples = sort(distances, dims=1)[1:k, 1:end]
    knn_graph = Matrix{Tuple{Int, Dtype}}(undef, k, nq)
    for i = 1:k, j = 1:nq
        knn_graph[i, j] = (knn_graph_tuples[i, j].idx, knn_graph_tuples[i, j].dist)
    end
    return getindex.(knn_graph, 1), getindex.(knn_graph, 2)
end

