module KNNGraphs

using Distances: PreMetric, SemiMetric, result_type, evaluate
using DataStructures: BinaryMaxHeap, top, pop!
using SparseArrays: sparse

# imports for adding method definitions
import Base: eltype, ==, <, isless
import LightGraphs: AbstractGraph, edges, edgetype, has_edge, has_vertex, inneighbors, ne, nv, add_edge!
import LightGraphs: outneighbors, vertices, is_directed, AbstractEdge, src, dst, reverse, weights


# Abstract KNN Graph definitions
"""
ApproximateKNNGraph{V, K} subtypes are weighted, directed graphs where each vertex
has exactly `k` forward edges.
"""
abstract type ApproximateKNNGraph{V, K, U <: Real} <: AbstractGraph{V} end

# interface
eltype(::ApproximateKNNGraph{V, K, U}) where {V, K, U} = V
# all knn graphs are directed
is_directed(::Type{<:ApproximateKNNGraph}) = true
is_directed(::ApproximateKNNGraph) = true

include("abstract.jl")
include("heap_edge.jl")
include("heap_graph.jl")
include("heap_utils.jl")

# export public interface
export ApproximateKNNGraph, HeapKNNGraph, HeapKNNGraphEdge, knn_diameter 

end
