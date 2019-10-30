# temporary draft implementation to eventually replace nn_descent.jl

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
    old_neighbors = [BitSet() for _ in 1:nv(graph)]
    new_neighbors = [BitSet() for _ in 1:nv(graph)]
    for e in edges(graph)
        if flag(e) # new edges hasn't participated in local join
            # mark sampled new forward neighbors as old (set flag to false)
            if rand() ≤ sample_rate
                e.flag = false
                union!(new_neighbors[src(e)], dst(e))
                union!(new_neighbors[dst(e)], src(e))
            end

        else # old neighbor
            # always include old fw neighbors
            union!(old_neighbors[src(e)], dst(e))
            # sample to include old reverse neighbors
            if rand() ≤ sample_rate
                union!(old_neighbors[dst(e)], src(e))
            end
        end
    end

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

"""
    nndescent(::Type{graph_type}, data, nneighbors, metric)

Find the approximate neighbors of each point in `data` by  iteratively
refining a KNN graph of type `graph_type`. Returns the final KNN graph.
"""
function nndescent(::Type{G}, 
                   data::D, 
                   nneighbors::Integer, 
                   metric::PreMetric;
                   max_iters = 10,
                   sample_rate = 1,
                   precision = 1//1000,
                  ) where {G <: ApproximateKNNGraph,
                           V <: AbstractVector,
                           D <: AbstractVector{V}}
    
    #validate_args(G, data, nneighbors, metric, max_iters, sample_rate, precision)

    graph = G(data, nneighbors, metric)
    for i in 1:max_iters
        c = local_join!(graph, data, metric; sample_rate=sample_rate)
        if c ≤ precision * nneighbors * nv(graph)
            break
        end
    end
    return graph
end
