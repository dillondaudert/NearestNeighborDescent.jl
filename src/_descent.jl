# temporary draft implementation to eventually replace nn_descent.jl

"""
    nndescent(data, n_neighbors, metric)

Find the approximate neighbors of each point in `data` by  iteratively
refining a KNN graph of type `graph_type`. Returns the final KNN graph.

# Arguments
- `max_iters = 10`: Limits the number of iterations to refine candidate
nearest neighbors. Higher values trade off speed for accuracy. Note that graph
construction may terminate early if little progress is being made.
- `sample_rate = 1`: The sample rate for calculating *local joins*
around each point. Lower values trade off accuracy for speed.
- `precision = 1e-3`: The threshold for early termination,
where precision is "roughly the fraction of true kNN allowed to be missed due to
early termination". Lower values take longer but return more accurate results.
"""
function nndescent(data::AbstractVector,
                   n_neighbors::Integer,
                   metric::PreMetric;
                   max_iters = 10,
                   sample_rate = 1,
                   precision = 1e-3,
                  ) where {G <: ApproximateKNNGraph}

    validate_args(data, n_neighbors, metric, max_iters, sample_rate, precision)

    graph = HeapKNNGraph(data, n_neighbors, metric)
    for i in 1:max_iters
        c = local_join!(graph, data, metric; sample_rate=sample_rate)
        if c ≤ precision * n_neighbors * nv(graph)
            break
        end
    end
    return graph
end

"""
    local_join!(g::HeapKNNGraph, data, metric::PreMetric; kwargs...)

Perform a local join on each vertex `v`'s neighborhood `N[v]` in `g`. Given vertex `v`
and its neighbors `N[v]`, compute the similar `metric(p, q)` for each pair `p, q ∈ N[v]` and
update `N[q]` and `N[p]`.

This mutates `g` in-place and returns a nonnegative integer indicating how many neighbor
updates took place during the local join.
"""
function local_join!(graph::HeapKNNGraph, data, metric::PreMetric; sample_rate = 1)
    # find in and out neighbors - old neighbors have already participated in a previous local join
    old_neighbors, new_neighbors = _get_neighbors(graph, sample_rate)
    c = 0
    # compute local join
    for v in vertices(graph)
        for p in new_neighbors[v]
            for q in (q_ for q_ in new_neighbors[v] if p < q_)
                # both new
                weight = evaluate(metric, data[p], data[q])
                c += add_edge!(graph, edgetype(graph)(p, q, weight))
                if !(metric isa SemiMetric) # not symmetric
                    weight = evaluate(metric, data[q], data[p])
                end
                c += add_edge!(graph, edgetype(graph)(q, p, weight))

            end
            for q in (q_ for q_ in old_neighbors[v] if p != q_)
                # one new, one old
                weight = evaluate(metric, data[p], data[q])
                c += add_edge!(graph, edgetype(graph)(p, q, weight))
                if !(metric isa SemiMetric) # not symmetric
                    weight = evaluate(metric, data[q], data[p])
                end
                c += add_edge!(graph, edgetype(graph)(q, p, weight))
            end
        end
    end

    return c
end


function local_join!(graph::LockHeapKNNGraph, data, metric::PreMetric; sample_rate = 1)
    old_neighbors, new_neighbors = _get_neighbors(graph, sample_rate)
    count = Threads.Atomic{Int}(0)
    # compute local join
    Threads.@threads for v in vertices(graph)
        for p in new_neighbors[v]
            for q in (q_ for q_ in new_neighbors[v] if p < q_)
                # both new
                weight = evaluate(metric, data[p], data[q])
                res = add_edge!(graph, edgetype(graph)(p, q, weight))
                Threads.atomic_add!(count, Int(res))
                if !(metric isa SemiMetric) # not symmetric
                    weight = evaluate(metric, data[q], data[p])
                end
                res = add_edge!(graph, edgetype(graph)(q, p, weight))
                Threads.atomic_add!(count, Int(res))
            end
            for q in (q_ for q_ in old_neighbors[v] if p != q_)
                # one new, one old
                weight = evaluate(metric, data[p], data[q])
                res = add_edge!(graph, edgetype(graph)(p, q, weight))
                Threads.atomic_add!(count, Int(res))
                if !(metric isa SemiMetric) # not symmetric
                    weight = evaluate(metric, data[q], data[p])
                end
                res = add_edge!(graph, edgetype(graph)(q, p, weight))
                Threads.atomic_add!(count, Int(res))
            end
        end
    end
    return count[]
end
