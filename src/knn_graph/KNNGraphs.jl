module KNNGraphs

using DataStructures
using Distances
using LightGraphs
using SparseArrays

include("abstract.jl")
include("heap_edge.jl")
include("threaded_heap_graph.jl")
include("heap_graph.jl")
include("heap_utils.jl")

# export public interface
export ApproximateKNNGraph, HeapKNNGraph, LockHeapKNNGraph, HeapKNNGraphEdge
export knn_diameter, knn_matrices
export flag, weight, edge_indices, node_edge, node_edges, update_flag!

end
