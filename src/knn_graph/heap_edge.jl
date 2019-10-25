
"""
    HeapKNNGraphEdge{T, U}

A weighted graph edge along with a flag
"""
mutable struct HeapKNNGraphEdge{T<:Integer, U<:Real} <: AbstractEdge{T}
    src::T
    dst::T
    weight::U
    flag::Bool
    function HeapKNNGraphEdge{T, U}(src, dst, weight, flag) where {T<:Integer, U<:Real}
        # assert no self edges
        src != dst || error("src cannot be the same as dst (no self loops) for HeapKNNGraphEdge")
        return new(src, dst, weight, flag)
    end
end
function HeapKNNGraphEdge(src::T, dst::T, weight::U, flag::Bool) where {T<:Integer, U<:Real}
    return HeapKNNGraphEdge{T, U}(src, dst, weight, flag)
end
HeapKNNGraphEdge(s, d, w) = HeapKNNGraphEdge(s, d, w, false)

function Base.:(==)(a::HeapKNNGraphEdge, b::HeapKNNGraphEdge)
    return src(a) == src(b) && dst(a) == dst(b) && weight(a) == weight(b) && flag(a) == flag(b)
end

Base.:(<)(a::HeapKNNGraphEdge, b::HeapKNNGraphEdge) = weight(a) < weight(b)

Base.isless(a::HeapKNNGraphEdge, b::HeapKNNGraphEdge) = a < b

flag(e::HeapKNNGraphEdge) = e.flag
weight(e::HeapKNNGraphEdge) = e.weight
# lightgraphs interface
LightGraphs.src(e::HeapKNNGraphEdge) = e.src
LightGraphs.dst(e::HeapKNNGraphEdge) = e.dst
LightGraphs.reverse(e::E) where {E <: HeapKNNGraphEdge} = E(dst(e), src(e), weight(e), flag(e))
