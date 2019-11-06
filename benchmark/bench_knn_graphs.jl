module BenchKNNGraphs

using NearestNeighborDescent.KNNGraphs
using Distances
using BenchmarkTools
using Random

suite = BenchmarkGroup()

knn_graph_types = [HeapKNNGraph]
dtypes = [Float32, Float64]
num_points = [50, 250, 1250, 6250]
dims = [5, 25, 125, 625]
num_neighbs = [5, 10, 20, 50, 100]
metrics = [SqEuclidean, CosineDist]

for graph in knn_graph_types
    for dtype in dtypes
        for np in num_points
            for d in dims
                data = [2 .* rand(dtype, d) .- 1 for _ in 1:np] # uniform dist in [-1, 1]
                for k in num_neighbs
                    for metric in metrics
                        prepath = [string(graph)]
                        postpath = [string(dtype), "N="*string(np), "D="*string(d), "K="*string(k), string(metric)]
                        suite[vcat(prepath, ["construction"], postpath)] = 
                            @benchmarkable $(graph)($(data), $k, $(metric()))
                        
                    end
                end
            end
        end
    end
end

end # module

BenchKNNGraphs.suite
