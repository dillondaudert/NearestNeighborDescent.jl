module BenchDescent

using NearestNeighborDescent
using NearestNeighborDescent: local_join!, get_neighbors!
using LightGraphs
using Distances
using BenchmarkTools
using Random

function two_local_join!(graph, data, metric, sample_rate)
    local_join!(graph, data, metric; sample_rate=sample_rate)
    local_join!(graph, data, metric; sample_rate=sample_rate)
end

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
                for rate in [1, .25]
                    prepath = ["N="*string(np), "D="*string(d), "K="*string(k), string(graph)]
                    postpath = [string(metric), string(rate)]
                    g = graph(data, k, metric())

                    suite[vcat(prepath, ["get_neighbors!"], postpath)] =
                        @benchmarkable get_neighbors!(g_, $rate) setup=(g_ = deepcopy($g))
                    g2 = graph(data, k, metric())
                    suite[vcat(prepath, ["local_join! x 2"], postpath)] =
                        @benchmarkable two_local_join!(g_, $(data), $(metric()), $(rate)) setup=(g_ = deepcopy($g2))
                end
            end
        end
    end
end

end # module

BenchDescent.suite
