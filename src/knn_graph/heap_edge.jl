
"""
    HeapKNNGraphEdge{T, U}

A weighted graph edge along with a flag
"""
mutable struct HeapKNNGraphEdge{T<:Integer, U<:Real} <: AbstractEdge{T}
    src::T
    dst::T
    weight::U
    flag::Bool
    function HeapKNNGraphEdge{T, U}(src, dst, weight, flag=true) where {T<:Integer, U<:Real}
        # assert no self edges
        src != dst || error("src cannot be the same as dst (no self loops) for HeapKNNGraphEdge")
        return new(src, dst, weight, flag)
    end
end
function HeapKNNGraphEdge(src::T, dst::T, weight::U, flag::Bool) where {T<:Integer, U<:Real}
    return HeapKNNGraphEdge{T, U}(src, dst, weight, flag)
end
HeapKNNGraphEdge(s, d, w) = HeapKNNGraphEdge(s, d, w, true)

@inline function Base.:(==)(a::HeapKNNGraphEdge{V, U}, b::HeapKNNGraphEdge{V, U}) where {V, U <: AbstractFloat}
    return src(a) == src(b) && dst(a) == dst(b) && isapprox(weight(a), weight(b); atol=eps(U))
end
@inline function Base.:(==)(a::HeapKNNGraphEdge{V, U}, b::HeapKNNGraphEdge{V, U}) where {V, U}
    return src(a) == src(b) && dst(a) == dst(b) && weight(a) == weight(b)
end

@inline Base.:(<)(a::HeapKNNGraphEdge, b::HeapKNNGraphEdge) = weight(a) < weight(b)

@inline flag(e::HeapKNNGraphEdge) = e.flag
@inline weight(e::HeapKNNGraphEdge) = e.weight
# lightgraphs interface
@inline LightGraphs.src(e::HeapKNNGraphEdge) = e.src
@inline LightGraphs.dst(e::HeapKNNGraphEdge) = e.dst
@inline LightGraphs.reverse(e::E) where {E <: HeapKNNGraphEdge} = E(dst(e), src(e), weight(e), flag(e))
