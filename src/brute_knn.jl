"""
Brute-force (𝒪(n²)) kNN search algorithm.
Returns an NxK array of the k nearest neighbors for each point in `data`.
"""
function brute_knn(data::Vector{V},
                   metric::Metric,
                   k::Int) where {V <: AbstractArray}

    np = length(data)
    distances = fill(_NNTuple(0, 0.), (np, np))

    for i = 1:np, j = 1:np
        d = evaluate(metric, data[i], data[j])
        distances[i, j] = _NNTuple(j, d)
    end

    # start at 2 as index 1 is the distance to itself (once sorted)
    knn = sort(distances, dims=2)[1:end,2:k+1]
    knn_ids = zeros(Int64, size(knn))
    for i = 1:np, j = 1:k
        knn_ids[i, j] = knn[i, j].idx
    end
    return knn_ids
end
