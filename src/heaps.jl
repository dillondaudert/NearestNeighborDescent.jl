# implementation of arrays as min-max heaps

using Base.Order: lt, Ordering, Forward, Reverse

function minmax_heapify!(A::AbstractVector)
    for i = length(A):-1:1
        trickledown!(A, i)
    end
end

"""

"""
function trickledown!(A::AbstractVector, i::Integer)

    if level(i) % 2 == 0
        trickledown!(A, i, Forward)
    else
        trickledown!(A, i, Reverse)
    end

end

function trickledown!(A::AbstractVector, i::Integer, o::Ordering, x=A[i])
    N = length(A)
    if lchild(i) ≤ N
        # tuple (min(val, j1) max(val, j2))
        # NOTE: this will find the index of the minimum or maximum descendant,
        #       depending on which ordering we're interested in
        ext = extrema((A[j], j) for j in descendants(N, i))
        m = lt(o, ext[1], ext[2]) ? ext[1][2] : ext[2][2]

        if m ≥ 4*i
            # this is a grandchild
            if lt(o, A[m], A[i])
                A[i] = A[m]
                A[m] = x
                if lt(o, A[hparent(m)], A[m])
                    t = A[m]
                    A[m] = A[hparent(m)]
                    A[hparent(m)] = t
                end
                trickledown!(A, m, o)
            end
        else
            if lt(o, A[m], A[i])
                A[i] = A[m]
                A[m] = x
            end
        end
    end
end

# utilities
@inline level(i) = round(Int, floor(log2(i)))
@inline lchild(i) = 2*i
@inline rchild(i) = 2*i+1
@inline children(i) = [lchild(i), rchild(i)]
@inline hparent(i) = round(Int, floor(i/2))

"""
Return the indices of all children and grandchildren of
position `i`.
"""
function descendants(N, i)
    children = [lchild(i), rchild(i)]
    grandchildren = []
    for child in children
        append!(grandchildren, [lchild(child), rchild(child)])
    end
    _descendants = vcat(children, grandchildren)
    return [d for d in _descendants if d ≤ N]
end

"""
    is_minmax_heap(A::AbstractVector) -> Bool

Return `true` if `A` is a min-max heap. A min-max heap is a
heap where the minimum element is the root and the maximum
element is a child of the root.
"""
function is_minmax_heap(A::AbstractVector)

    isheap = true
    N = length(A)

    for i = 1:N
        if level(i)%2 == 0
            # min layer
            # check that A[i] < children A[i]
            #    and grandchildren A[i]
            for j in descendants(N, i)
                isheap &= A[i] ≤ A[j]
            end
        else
            # max layer
            for j in descendants(N, i)
                isheap &= A[i] ≥ A[j]
            end
        end
    end
    isheap
end
