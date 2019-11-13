# threaded heap knn graph

"""
LockHeapKNNGraph - uses locks to synchronize the heaps that store the underlying
graph edge data. The heaps themselves are *not* thread-safe.
"""
struct LockHeapKNNGraph{V<:Integer, K, U<:Real} <: ApproximateKNNGraph{V, K, U}
    _knn_heaps::Vector{BinaryMaxHeap{HeapKNNGraphEdge{V, U}}}
    _heap_locks::Vector{ReentrantLock}
end
function LockHeapKNNGraph(data::D, k::Integer, metric::PreMetric) where {V, D <: AbstractVector{V}}

    # assert some invariants
    k > 0 || error("LockHeapKNNGraph needs positive `k`")
    length(data) > k || error("LockHeapKNNGraph needs `length(data) > k`")

    np = length(data)
    U = result_type(metric, first(data), first(data))
    HeapType = BinaryMaxHeap{HeapKNNGraphEdge{typeof(k), U}}
    knn_heaps = HeapType[HeapType() for _ in 1:np]
    heap_locks = ReentrantLock[ReentrantLock() for _ in 1:np]
    # initialize approx knn heaps randomly
    Threads.@threads for i in eachindex(data)
        k_idxs = sample_neighbors(np, k, exclude=[i])
        for j in k_idxs
            weight = evaluate(metric, data[i], data[j])
            lock(heap_locks[i]) do
                _heappush!(knn_heaps[i], HeapKNNGraphEdge(i, j, weight), k)
            end
            if !(metric isa SemiMetric)
                weight = evaluate(metric, data[j], data[i])
            end
            lock(heap_locks[j]) do
                _heappush!(knn_heaps[j], HeapKNNGraphEdge(j, i, weight), k)
            end
        end
    end
    return LockHeapKNNGraph{typeof(k), k, U}(knn_heaps, heap_locks)
end

"""
    inneighbors(g::LockHeapKNNGraph, v)

Similar to `inneighbors(g::HeapKNNGraph, v)`, except it acquires locks for the neighbor heaps
as it iterates in order for this to be thread-safe.

"""
@inline function LightGraphs.inneighbors(g::HeapKNNGraph{V}, v::V) where V
    neighbs = V[]
    for i in nv(g)
        lock(g._heap_locks[i]) do
            for e in g._knn_heaps[i].valtree
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
@inline function LightGraphs.outneighbors(g::HeapKNNGraph{V}, v::V) where V
    neighbs = lock(g._heap_locks[v]) do
        dst.(g._knn_heaps[v].valtree)
    end
    return neighbs
end

"""
    add_edge!(g::LockHeapKNNGraph, e::HeapKNNGraphEdge)

Similar to `add_edge!(g::HeapKNNGraph, e)`, but made thread-safe using locks.
"""
function LightGraphs.add_edge!(g::LockHeapKNNGraph, e::HeapKNNGraphEdge)
    # NOTE we can assume the invariants for heap knn graphs hold
    lock(g._heap_locks[src(e)]) do
        if e < top(g._knn_heaps[src(e)]) && !has_edge(g, src(e), dst(e))
            # we know this edge is smaller than the top, so we can start by removing that
            pop!(g._knn_heaps[src(e)])
            push!(g._knn_heaps[src(e)], e)
            return true
        end
        return false
    end
end

"""
    node_edges(graph::LockHeapKNNGraph, i) -> edges

Return all the outgoing edges from node i in an arbitrary order. Thread-safe.
"""
@inline function node_edges(g::LockHeapKNNGraph{V, K}, i::V) where {V, K}
    lock(g._heap_locks[i]) do
        return edgetype(g)[node_edge(g, i, j) for j in one(V):K]
    end
end

"""
    update_flag!(g::LockHeapKNNGraph, i, j, flag)

Update the flag of the edge at the given indices. Since the flags don't influence
the edge ordering, this can't invalidate the heap invariant. Uses locks to ensure
thread safety.
"""
function update_flag!(g::LockHeapKNNGraph{V}, i::V, j::V, flag::Bool) where V
    lock(g._heap_locks[i]) do
        edge = node_edge(g, i, j)
        newedge = edgetype(g)(src(edge), dst(edge), weight(edge), flag)
        g._knn_heaps[i].valtree[j] = newedge
        return newedge
    end
end
