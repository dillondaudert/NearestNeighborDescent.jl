# temporary draft implementation to eventually replace nn_descent.jl

"""
    nndescent(::Type{graph_type}, data, nneighbors, metric)

Find the approximate neighbors of each point in `data` by  iteratively
refining a KNN graph of type `graph_type`. Returns the final KNN graph.
"""
function nndescent(::Type{G}, data, nneighbors, metric) where {G <: ApproximateKNNGraph}
    graph = G(data, nneighbors, metric)
    while true
        ns = [neighbors(graph, v) for v in vertices(graph)]
        c = 0
        for v in vertices(graph)
            for p in ns[v], q in ns[v]
                if p == q
                    continue
                end
                w = evaluate(metric, data[p], data[q])
                c = c + add_edge!(graph, edgetype(graph)(p, q, w, false))
            end
        end
        if c == 0
            break
        end
    end
    return graph
end
