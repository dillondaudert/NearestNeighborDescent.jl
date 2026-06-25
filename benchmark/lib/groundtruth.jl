# Exact KNN ground truth and recall.
#
# An approximate algorithm cannot be benchmarked on speed alone: the whole point
# is the speed-vs-accuracy trade-off (ann-benchmarks plots recall@k against QPS).
# These helpers provide the exact neighbors needed to score recall. For synthetic
# data we compute ground truth by brute force; for the real ann-benchmarks HDF5
# datasets the ground truth ships in the file (see datasets.jl).

"""
    bruteforce_knn(base, queries, k, metric) -> idx::Matrix{Int}

Exact `k` nearest neighbors of each query among `base` (both
`Vector{<:AbstractVector}`). Returns a `k × length(queries)` matrix of 1-based
indices into `base`, sorted nearest-first. Used to score recall for synthetic
data and to self-check the harness against the shipped ground truth.
"""
function bruteforce_knn(base::AbstractVector,
                        queries::AbstractVector,
                        k::Integer,
                        metric::PreMetric)
    nb = length(base)
    k ≤ nb || error("k=$k exceeds base size $nb")
    idx = Matrix{Int}(undef, k, length(queries))
    dists = Vector{Float64}(undef, nb)
    order = Vector{Int}(undef, nb)
    for (qi, q) in enumerate(queries)
        @inbounds for bi in 1:nb
            dists[bi] = evaluate(metric, q, base[bi])
        end
        sortperm!(order, dists)
        @inbounds for j in 1:k
            idx[j, qi] = order[j]
        end
    end
    return idx
end

"""
    bruteforce_self_knn(data, k, metric) -> idx::Matrix{Int}

Exact `k` nearest neighbors of each point among the other points (excludes self),
matching the semantics of an NNDescent graph over `data`.
"""
function bruteforce_self_knn(data::AbstractVector, k::Integer, metric::PreMetric)
    full = bruteforce_knn(data, data, k + 1, metric)   # +1 because self is nearest
    out = Matrix{Int}(undef, k, length(data))
    for i in 1:length(data)
        c = 0
        for j in 1:(k + 1)
            full[j, i] == i && continue
            c += 1
            c > k && break
            out[c, i] = full[j, i]
        end
        # if self was not in the top k+1 (ties), backfill is unnecessary in practice
        c < k && (out[c+1:k, i] .= full[k+1, i])
    end
    return out
end

"""
    recall(approx_idx, truth_idx) -> Float64

Mean recall@k over all queries: the fraction of each query's true `k` neighbors
that appear in the approximate result. `approx_idx` and `truth_idx` are
`k × nqueries` (or `≥k × nqueries`) index matrices; recall uses
`k = size(truth_idx, 1)`. This is the standard ann-benchmarks recall.
"""
function recall(approx_idx::AbstractMatrix, truth_idx::AbstractMatrix)
    k, q = size(truth_idx)
    size(approx_idx, 2) == q || error("query count mismatch")
    size(approx_idx, 1) ≥ k || error("approx has fewer than k neighbors")
    total = 0
    truthset = Set{Int}()
    for j in 1:q
        empty!(truthset)
        for i in 1:k
            push!(truthset, truth_idx[i, j])
        end
        for i in 1:size(approx_idx, 1)
            approx_idx[i, j] in truthset && (total += 1)
        end
    end
    return total / (k * q)
end
