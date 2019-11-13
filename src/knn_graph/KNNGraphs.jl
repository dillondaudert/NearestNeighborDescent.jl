module KNNGraphs

using DataStructures
using Distances
using LightGraphs
using SparseArrays

# Abstract KNN Graph definitions
"""
ApproximateKNNGraph{V, K} subtypes are weighted, directed graphs where each vertex
has exactly `k` forward edges.
"""
abstract type ApproximateKNNGraph{V, K, U <: Real} <: AbstractGraph{V} end

# interface
Base.eltype(::ApproximateKNNGraph{V, K, U}) where {V, K, U} = V
# all knn graphs are directed
LightGraphs.is_directed(::Type{<:ApproximateKNNGraph}) = true
LightGraphs.is_directed(::ApproximateKNNGraph) = true

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
