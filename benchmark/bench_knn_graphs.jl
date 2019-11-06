module BenchKNNGraphs

using NearestNeighborDescent.KNNGraphs
using LightGraphs
using Distances
using BenchmarkTools
using Random

function graph_has_edges(graph, es)
    for e in es
        has_edge(graph, e)
    end
end

suite = BenchmarkGroup()

knn_graph_types = [HeapKNNGraph]
dtypes = [Float32, Float64]
num_points = [1000]
dims = [50]
num_neighbs = [20]
metrics = [SqEuclidean]

for graph in knn_graph_types
    for dtype in dtypes
        for np in num_points
            for d in dims
                Random.seed!(0)
                data = [2 .* rand(dtype, d) .- 1 for _ in 1:np] # uniform dist in [-1, 1]
                for k in num_neighbs
                    if k ≥ np
                        continue
                    end
                    prepath = [string(graph)]
                    postpath = [string(dtype), "N="*string(np), "D="*string(d), "K="*string(k)]
                    for metric in metrics
                        suite[vcat(prepath, ["construction"], postpath, [string(metric)])] = 
                            @benchmarkable $(graph)($(data), $k, $(metric()))

                        g = graph(data, k, metric())
                        es = shuffle!(collect(edges(g)))[1:k]

                        suite[vcat(prepath, ["has_edge true"], postpath, [string(metric)])] = 
                            @benchmarkable graph_has_edges($g, $es)

                        es2 = similar(es)
                        for e in es
                            _e = deepcopy(e)
                            _e.weight = rand(typeof(weight(e)))
                            push!(es2, _e)
                        end

                        suite[vcat(prepath, ["has_edge false"], postpath, [string(metric)])] =
                            @benchmarkable graph_has_edges($g, $es2)
                    end
                end
            end
        end
    end
end

end # module

BenchKNNGraphs.suite
