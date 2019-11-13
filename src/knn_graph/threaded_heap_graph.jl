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
