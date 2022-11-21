
"""
    HeapKNNGraphEdge{T, U}

A weighted graph edge along with a flag
"""
struct HeapKNNGraphEdge{T<:Integer, U<:Real} <: AbstractEdge{T}
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
    return src(a) == src(b) && dst(a) == dst(b) && isapprox(weight(a), weight(b))
end
@inline function Base.:(==)(a::HeapKNNGraphEdge{V, U}, b::HeapKNNGraphEdge{V, U}) where {V, U}
    return src(a) == src(b) && dst(a) == dst(b) && weight(a) == weight(b)
end

@inline function Base.:(<)(a::HeapKNNGraphEdge{V, U}, b::HeapKNNGraphEdge{V, U}) where {V, U<:AbstractFloat}
    # for floats, check that a<b AND that they aren't approximately equal
    return weight(a) < weight(b) && !isapprox(weight(a), weight(b))
end
@inline Base.:(<)(a::HeapKNNGraphEdge{V, U}, b::HeapKNNGraphEdge{V, U}) where {V, U} = weight(a) < weight(b)
Base.isless(a::HeapKNNGraphEdge, b::HeapKNNGraphEdge) = a < b

Base.eltype(e::HeapKNNGraphEdge{V}) where V = V

@inline flag(e::HeapKNNGraphEdge) = e.flag
@inline weight(e::HeapKNNGraphEdge) = e.weight
# graphs interface
@inline Graphs.src(e::HeapKNNGraphEdge) = e.src
@inline Graphs.dst(e::HeapKNNGraphEdge) = e.dst
@inline Graphs.reverse(e::E) where {E <: HeapKNNGraphEdge} = E(dst(e), src(e), weight(e), flag(e))
