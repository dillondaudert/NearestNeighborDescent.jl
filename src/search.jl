

"""
    search(graph, queries, n_neighbors, metric; max_candidates) -> indices, distances

Search the kNN `graph` for the nearest neighbors of the points in `queries`.
`max_candidates` controls how large the candidate queue should be (min `n_neighbors`);
larger values increase accuracy at the cost of speed.
"""
function search(graph::G,
                data::D,
                queries::D,
                n_neighbors::Integer,
                metric::PreMetric;
                max_candidates=max(n_neighbors, 20),
                ) where {V, K, U, G <: ApproximateKNNGraph{V, K, U}, D <: AbstractVector}

    length(queries) ≥ 1 || error("queries must have at least 1 element")
    n_neighbors ≥ 1 || error("n_neighbors must be at least 1")
    max_candidates ≥ 5 || error("max_candidates must be at least 5")

    # lists of candidates, sorted by distance
    candidates = [BinaryMaxHeap{Tuple{U, V, Bool}}() for _ in 1:length(queries)]
    # a set of seen candidates per thread
    seen_sets = [BitVector(undef, length(data)) for _ in 1:Threads.nthreads()]
    Threads.@threads for i in eachindex(queries)
        # zero out seen
        seen = seen_sets[Threads.threadid()]
        seen .= false
        # initialize with random
        init_candidates!(candidates[i], seen, graph, data, queries[i], metric, max_candidates)
        while true
            next_candidate = get_next_candidate!(candidates[i])
            if isnothing(next_candidate)
                break
            end

            for v in outneighbors(graph, next_candidate[2])
                if !seen[v]
                    dist = evaluate(metric, queries[i], data[v])
                    if dist ≤ top(candidates[i])[1]
                        pop!(candidates[i]) # pop maximum
                        push!(candidates[i], (dist, v, false))
                    end
                    seen[v] = true
                end
            end
        end
    end

    return deheap_knns(candidates, n_neighbors)
end

function search(graph::G,
                data::D,
                queries::D,
                n_neighbors::Integer,
                metric::PreMetric;
                max_candidates=max(n_neighbors, 20),
                ) where {V, K, U, G <: ApproximateKNNGraph{V, K, U}, D <: AbstractMatrix}
    data_cols = [col for col in eachcol(data)]
    query_cols = [col for col in eachcol(queries)]
    return search(graph, data_cols, query_cols, n_neighbors, metric; max_candidates=max_candidates)
end


function get_next_candidate!(candidates::BinaryMaxHeap{Tuple{U, V, Bool}}) where {U <: Real, V}
    min_idx = -1
    min_dist = typemax(U)
    for (i, t) in enumerate(candidates.valtree)
        if t[1] < min_dist && !t[3]
            min_idx = i
            min_dist = t[1]
        end
    end
    if min_idx != -1 # found an unvisited candidate
        dist, node, _ = candidates.valtree[min_idx]
        cand = (dist, node, true)
        candidates.valtree[min_idx] = cand # mark visited
        return cand
    end
    return nothing
end


function init_candidates!(candidates, seen, graph, data, query, metric, max_candidates)
    for v in KNNGraphs.sample_neighbors(nv(graph), max_candidates)
        dist = evaluate(metric, query, data[v])
        push!(candidates, (dist, v, false))
        seen[v] = true
    end
    return candidates
end

"""
Remove the `k` nearest neighbors from each heap in `knn_heaps`.
Return two k x length(knn_heaps) arrays for the indices and
distances to each point's kNN.
"""
function deheap_knns(heaps::Vector{BinaryMaxHeap{Tuple{U, V, Bool}}}, k) where {U, V}

    ids = Array{V}(undef, (k, length(heaps)))
    dists = Array{U}(undef, (k, length(heaps)))

    for i in 1:length(heaps)
        len = length(heaps[i])
        for j in 1:len
            # NOTE: these are max heaps, so we only want the last k
            node_dist, node_idx, _ = pop!(heaps[i])
            neighbor_idx = 1 + len - j
            if neighbor_idx <= k
                ids[neighbor_idx, i] = node_idx
                dists[neighbor_idx, i] = node_dist
            end
        end
    end
    return ids, dists
end
