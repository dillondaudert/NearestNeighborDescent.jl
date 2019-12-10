module BenchKNNGraphs

using NearestNeighborDescent.KNNGraphs
using LightGraphs
using Distances
using BenchmarkTools
using Random

#function graph_has_edges(graph, _edges)
#    for e in _edges
#        has_edge(graph, e)
#    end
#end

suite = BenchmarkGroup()

knn_graph_types = [HeapKNNGraph, LockHeapKNNGraph]
num_points = [500]
dims = [50]
num_neighbs = [15]
metric = Euclidean

for graph in knn_graph_types
    for np in num_points
        for d in dims
            Random.seed!(0)
            data = [2 .* rand(d) .- 1 for _ in 1:np] # uniform dist in [-1, 1]
            for k in num_neighbs
                if k ≥ np
                    continue
                end
                prepath = ["N="*string(np), "D="*string(d), "K="*string(k), string(graph)]
                postpath = [string(metric)]
                suite[vcat(prepath, ["construction"], postpath, )] =
                    @benchmarkable $(graph)($(data), $k, $(metric()))
            end
        end
    end
end

end # module

BenchKNNGraphs.suite
