# temporary draft implementation to eventually replace nn_descent.jl


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
        old_ns, new_ns = KNNGraphs._all_neighbors(graph)
        c = 0
        # DO LOCAL JOIN
        for v in vertices(graph)
            for p in new_ns[v]
                for q in new_ns[v]
                    # both new
                    if p < q
                        w = evaluate(metric, data[p], data[q])
                        c = c + add_edge!(graph, edgetype(graph)(p, q, w, false))
                    end
                end
                for q in old_ns[v]
                    # one new, one old
                    if p < q
                        w = evaluate(metric, data[p], data[q])
                        c = c + add_edge!(graph, edgetype(graph)(p, q, w, false))
                    end
                end
            end
        end
        if c ≤ precision * nneighbors * nv(graph)
            break
        end
    end
    return graph
end
