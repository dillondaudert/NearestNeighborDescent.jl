

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
    candidates = [BinaryMinHeap{Tuple{U, V, Bool}}() for _ in 1:length(queries)]
    seen = BitVector(undef, length(data))
    for i in eachindex(queries)
        # zero out seen
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
    return candidates
end


function get_next_candidate!(candidates::BinaryMinHeap{Tuple{U, V, Bool}}) where {U <: Real, V}
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
