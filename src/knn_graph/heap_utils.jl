
"""
    _heappush!(heap::BinaryMaxHeap, edge::HeapKNNGraphEdge, max_length)

Try to push `edge` to `heap`. This will fail if `edge` is
already in `heap`.
If `length(heap) > max_length` after pushing, `pop` the largest edge off.
"""
function _heappush!(heap::BinaryMaxHeap,
                    edge::HeapKNNGraphEdge,
                    max_length::Integer)

    # edge not already in heap
    if !(edge in heap.valtree)
        push!(heap, edge)
    end

    if length(heap) > max_length
        pop!(heap)
    end

    return
end

"""
Sample `n_neighbors` elements from a set of ints `1:npoints`.
The ints in `exclude` won't be sampled.
"""
function sample_neighbors(npoints::Int,
                          n_neighbors::Int,
                          sample_rate::R = 1.;
                          exclude::Vector{Int} = Vector{Int}()) where {R <: AbstractFloat}
    # num neighbors to sample = max(ρK, N)
    last = min(npoints-length(exclude), trunc(Int, sample_rate*n_neighbors))
    ids = Set{Int}()
    while length(ids) < last
        k = (abs(rand(Int)) % npoints) + 1
        if k ∉ exclude
            union!(ids, k)
        end
    end
    return collect(ids)
end
