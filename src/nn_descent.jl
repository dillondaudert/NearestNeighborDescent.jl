# A simple NN graph implementation

struct DescentGraph{V <: AbstractVector,M,S <: Tuple}
    data::Vector{V}
    metric::M
    graph::Matrix{S}
end

"""
    DescentGraph(data::Vector{V}, n_neighbors::Integer[, metric::SemiMetric = Euclidean()]; <keyword arguments>)

Build an approximate kNN graph of `data` using nearest neighbor descent.
# Arguments
- `max_iters::Integer = 10`: Limits the number of iterations to refine candidate
nearest neighbors. Higher values trade off speed for accuracy. Note that graph
construction may terminate early if little progress is being made.
- `sample_rate::AbstractFloat = 1.`: The sample rate for calculating *local joins*
around each point. Lower values trade off accuracy for speed.
- `precision::AbstractFloat = .001`: The threshold for early termination,
where precision is "roughly the fraction of true kNN allowed to be missed due to
early termination". Lower values take longer but return more accurate results.
"""
function DescentGraph(data::Vector{V},
                      n_neighbors::Integer,
                      metric::SemiMetric = Euclidean();
                      max_iters::Integer = 10,
                      sample_rate::R = 1.,
                      precision::R = 0.001
                     ) where {V <: AbstractVector, R <: AbstractFloat}
    length(data) >= 2 || error("data must contain at least 2 elements")
    n_neighbors <= length(data) - 1 || error("n_neighbors must be 1 less than length(data)=", length(data))
    max_iters >= 1 || error("max_iters must be greater than 0")
    0. < sample_rate ≤ 1. || error("sample_rate must be in (0., 1.]")
    0. ≤ precision ≤ 1. || error("precision must be in [0., 1.]")
    DescentGraph(data,
                 metric,
                 build_graph(data, n_neighbors, metric, max_iters, sample_rate, precision))
end

"""
    DescentGraph(data::AbstractMatrix, n_neighbors::Integer[, metric::SemiMetric = Euclidean()]; <keyword arguments>)
"""
function DescentGraph(data::AbstractMatrix,
                      n_neighbors::Integer,
                      metric::SemiMetric = Euclidean();
                      max_iters::Integer = 10,
                      sample_rate::AbstractFloat = 1.,
                      precision::AbstractFloat = 0.001
                     )
    data_vectors = [@view data[:, i] for i in 1:size(data, 2)]
    DescentGraph(data_vectors, n_neighbors, metric; max_iters=max_iters, sample_rate=sample_rate, precision=precision)
end

"""
Return a kNN graph for the input data according to the given metric.
"""
function build_graph(data::Vector{V},
                     k::Integer,
                     metric::M,
                     max_iters::Integer,
                     sample_rate::R,
                     precision::R
                    ) where {V <: AbstractArray,
                             M <: SemiMetric,
                             R <: AbstractFloat}

    np = length(data)
    # initialize with random neighbors
    Dtype = result_type(metric, data[1], data[1])
    knn_heaps = make_knn_heaps(data, k, metric)

    # until no further updates
    for i = 1:max_iters
        # get the fw and bw neighbors of each point
        old_fw, fw, old_bw, bw = _neighbors(knn_heaps, sample_rate)
        old_neighbors = [union(old_fw[i], old_bw[i]) for i in 1:np]
        new_neighbors = [union(fw[i], bw[i]) for i in 1:np]
        c = 0
        # calculate local join around each point
        for i in 1:np
            for u₁ ∈ new_neighbors[i]
                for u₂ ∈ new_neighbors[i]
                    # both points are new
                    if i ≠ u₁ && i ≠ u₂ && u₁ < u₂
                        d = evaluate(metric, data[u₁], data[u₂])
                        c += _heappush!(knn_heaps[u₁], NNTuple(u₂, d))
                        c += _heappush!(knn_heaps[u₂], NNTuple(u₁, d))
                    end
                end
                for u₂ ∈ old_neighbors[i]
                    # one point is new
                    if i ≠ u₁ && i ≠ u₂
                        d = evaluate(metric, data[u₁], data[u₂])
                        c += _heappush!(knn_heaps[u₁], NNTuple(u₂, d))
                        c += _heappush!(knn_heaps[u₂], NNTuple(u₁, d))
                    end
                end
            end

        end
        if c <= precision*k*np
            break
        end
    end
    knn_graph = Matrix{Tuple{Int, Dtype}}(undef, k, np)
    for j in 1:np
        rev_nns = [pop!(knn_heaps[j]) for _ in 1:length(knn_heaps[j])]
        for i in 1:k
            knn_graph[i, j] = (rev_nns[end-(i-1)].idx, rev_nns[end-(i-1)].dist)
        end
    end
    return knn_graph
end

"""
    search(graph::DescentGraph, queries::Vector{V}, n_neighbors::Integer, queue_size=1.) -> indices, distances

Search the kNN `graph` for the nearest neighbors of the points in `queries`.
`queue_size` controls how large the candidate queue should be as a multiple of
`n_neighbors`. Larger values increase accuracy at the cost of speed.
"""
function search(graph::DescentGraph,
                queries::Vector{V},
                n_neighbors::Integer,
                queue_size::Real = 1.,
                ) where {V <: AbstractArray}
    length(queries) ≥ 1 || error("queries must have at least 1 element")
    n_neighbors ≥ 1 || error("n_neighbors must be at least 1")
    queue_size ≥ 1. || error("queue_size must be at least 1.")

    max_candidates = trunc(Int, n_neighbors*queue_size)
    Dtype = result_type(graph.metric, queries[1], queries[1])
    candidates = [binary_maxheap(NNTuple{Int, Dtype}) for _ in 1:length(queries)]
    for i in eachindex(queries)
        # init
        seen = fill(false, length(graph.data))
        j =  rand(1:length(graph.data))
        d = evaluate(graph.metric, queries[i], graph.data[j])
        _heappush!(candidates[i], NNTuple(j, d, false), max_candidates)
        seen[j] = true

        while true
            unexp = min_flagged(candidates[i])
            if unexp.idx == -1
                break
            end
            # expand closest unexpanded neighbor
            unexp.flag = true
            for t in graph.graph[:,unexp.idx]
                if !seen[t[1]]
                    seen[t[1]] = true
                    d = evaluate(graph.metric,
                                 queries[i],
                                 graph.data[t[1]])
                    _unchecked_heappush!(candidates[i],
                                         NNTuple(t[1], d, false),
                                         max_candidates)
                end
            end
        end
    end
    knn_graph = [[pop!(candidates[i]) for _ in 1:length(candidates[i])][end:-1:end-(n_neighbors-1)]
                 for i in 1:length(candidates)]
    ids = Array{Int}(undef, (n_neighbors, length(queries)))
    dists = Array{Dtype}(undef, (n_neighbors, length(queries)))

    for i = 1:length(queries), j in 1:n_neighbors
        ids[j, i] = knn_graph[i][j].idx
        dists[j, i] = knn_graph[i][j].dist
    end
    return ids, dists
end

"""
    search(graph::DescentGraph, queries::AbstractMatrix, n_neighbors::Integer, queue_size::Real = 1.)
"""
function search(graph::DescentGraph,
                queries::AbstractMatrix,
                n_neighbors::Integer,
                queue_size::Real = 1.,
                )
    queries_array = [@view queries[:, i] for i in 1:size(queries, 2)]
    search(graph, queries_array, n_neighbors, queue_size)
end