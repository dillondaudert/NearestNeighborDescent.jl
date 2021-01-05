# threaded heap knn graph

"""
LockHeapKNNGraph - uses locks to synchronize the heaps that store the underlying
graph edge data. The heaps themselves are *not* thread-safe.
"""
struct LockHeapKNNGraph{V<:Integer,
                        U<:Real,
                        D<:AbstractVector,
                        M<:PreMetric} <: ApproximateKNNGraph{V, U, D, M}
    heaps::Vector{BinaryMaxHeap{HeapKNNGraphEdge{V, U}}}
    locks::Vector{ReentrantLock}
    data::D
    n_neighbors::V
    metric::M
end
function LockHeapKNNGraph(data::D,
                          metric::M,
                          indices::AbstractMatrix{V},
                          distances::AbstractMatrix{U}) where {D <: AbstractVector,
                                                               M <: PreMetric,
                                                               V <: Integer,
                                                               U <: Real}
    n_neighbors = size(indices, 1)
    n_points = size(indices, 2)
    length(data) == size(indices, 2) || error("`indices` must have num columns equal to `length(data)`")
    size(indices) == size(distances) || error("`indices` and `distances` must have same shape")
    n_neighbors < n_points || error("`Must have more columns than rows`")
    # ...
    HeapType = BinaryMaxHeap{HeapKNNGraphEdge{V, U}}
    knn_heaps = HeapType[HeapType() for _ in 1:n_points]
    heap_locks = ReentrantLock[ReentrantLock() for _ in 1:n_points]
    for v in 1:n_points, i in 1:n_neighbors
        _heappush!(knn_heaps[v], HeapKNNGraphEdge(v, indices[i, v], distances[i, v]), n_neighbors)
    end
    return LockHeapKNNGraph{V, U, D, M}(knn_heaps, heap_locks, data, n_neighbors, metric)
end

function LockHeapKNNGraph(data::D, n_neighbors::Integer, metric::M) where {D <: AbstractVector,
                                                                           M <: PreMetric}

    # assert some invariants
    n_neighbors > 0 || error("LockHeapKNNGraph needs positive `n_neighbors`")
    length(data) > n_neighbors || error("LockHeapKNNGraph needs `length(data) > n_neighbors`")

    n_points = length(data)
    U = result_type(metric, first(data), first(data))
    HeapType = BinaryMaxHeap{HeapKNNGraphEdge{typeof(n_neighbors), U}}
    knn_heaps = HeapType[HeapType() for _ in 1:n_points]
    heap_locks = ReentrantLock[ReentrantLock() for _ in 1:n_points]
    # initialize approx knn heaps randomly
    Threads.@threads for i in eachindex(data)
        k_idxs = sample_neighbors(n_points, n_neighbors, exclude=[i])
        for j in k_idxs
            weight = evaluate(metric, data[i], data[j])
            lock(heap_locks[i]) do
                _heappush!(knn_heaps[i], HeapKNNGraphEdge(i, j, weight), n_neighbors)
            end
            if !(metric isa SemiMetric)
                weight = evaluate(metric, data[j], data[i])
            end
            lock(heap_locks[j]) do
                _heappush!(knn_heaps[j], HeapKNNGraphEdge(j, i, weight), n_neighbors)
            end
        end
    end
    return LockHeapKNNGraph{typeof(n_neighbors), U, D, M}(knn_heaps, heap_locks, data, n_neighbors, metric)
end

# convert matrix data input to vector of vectors (views)
function LockHeapKNNGraph(data::D, args...; kwargs...) where D <: AbstractMatrix
    return LockHeapKNNGraph(collect(eachcol(data)), args...; kwargs...)
end

"""
    inneighbors(g::LockHeapKNNGraph, v)

Similar to `inneighbors(g::HeapKNNGraph, v)`, except it acquires locks for the neighbor heaps
as it iterates in order for this to be thread-safe.

"""
@inline function LightGraphs.inneighbors(g::LockHeapKNNGraph{V}, v::V) where V
    neighbs = V[]
    for i in nv(g)
        lock(g.locks[i]) do
            for e in g.heaps[i].valtree
                if dst(e) == v
                    push!(neighbs, src(e))
                end
            end
        end
    end
    return neighbs
end

"""
    outneighbors(g::LockHeapKNNGraph, v)

Similar to `outneighbors(g::HeapKNNGraph, v)`, except locks the neighbor heap before collecting
to make this thread-safe.
"""
@inline function LightGraphs.outneighbors(g::LockHeapKNNGraph{V}, v::V) where V
    neighbs = lock(g.locks[v]) do
        dst.(g.heaps[v].valtree)
    end
    return neighbs
end

"""
    add_edge!(g::LockHeapKNNGraph, e::HeapKNNGraphEdge)

Similar to `add_edge!(g::HeapKNNGraph, e)`, but made thread-safe using locks.
"""
function LightGraphs.add_edge!(g::LockHeapKNNGraph, e::HeapKNNGraphEdge)
    # NOTE we can assume the invariants for heap knn graphs hold
    lock(g.locks[src(e)]) do
        if e < first(g.heaps[src(e)]) && !has_edge(g, src(e), dst(e))
            # we know this edge is smaller than the first, so we can start by removing that
            pop!(g.heaps[src(e)])
            push!(g.heaps[src(e)], e)
            return true
        end
        return false
    end
end

"""
    node_edges(graph::LockHeapKNNGraph, i) -> edges

Return all the outgoing edges from node i in an arbitrary order. Thread-safe.
"""
@inline function node_edges(g::LockHeapKNNGraph{V}, i::V) where V
    lock(g.locks[i]) do
        return edgetype(g)[node_edge(g, i, j) for j in one(V):g.n_neighbors]
    end
end

"""
    update_flag!(g::LockHeapKNNGraph, i, j, flag)

Update the flag of the edge at the given indices. Since the flags don't influence
the edge ordering, this can't invalidate the heap invariant. Uses locks to ensure
thread safety.
"""
function update_flag!(g::LockHeapKNNGraph{V}, i::V, j::V, flag::Bool) where V
    lock(g.locks[i]) do
        edge = node_edge(g, i, j)
        newedge = edgetype(g)(src(edge), dst(edge), weight(edge), flag)
        g.heaps[i].valtree[j] = newedge
        return newedge
    end
end
