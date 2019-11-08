# to eventually replace utils.jl

"""
Validate the args and kwargs for nndescent.
"""
function validate_args(data, n_neighbors, metric, max_iters, sample_rate, precision)
    max_iters >= 1 || error("max_iters must be greater than 0")
    0. < sample_rate ≤ 1. || error("sample_rate must be in (0., 1.]")
    0. ≤ precision ≤ 1. || error("precision must be in [0., 1.]")
    return
end

"""
Get the neighbors of each point in a KNN graph `graph` as sets of integer ids. 

For the NNDescent algorithm, these are separated into the old and new neighbors.
"""
function _get_neighbors(graph::HeapKNNGraph{V}, sample_rate) where V
    old_neighbors = [V[] for _ in 1:nv(graph)]
    new_neighbors = [V[] for _ in 1:nv(graph)]
    for ind in edge_indices(graph)
        @inbounds e = get_edge(graph, ind[1], ind[2])
        if flag(e) # isnew(e) => new edges haven't participated in local join
            if rand() ≤ sample_rate 
                # mark sampled new forward neighbors as old
                @inbounds e = update_flag!(graph, ind[1], ind[2], false)
                push!(new_neighbors[src(e)], dst(e))
                push!(new_neighbors[dst(e)], src(e))
            end
        else # old neighbors
            # always include old forward
            push!(old_neighbors[src(e)], dst(e))
            # sample old reverse neighbors
            if rand() ≤ sample_rate
                push!(old_neighbors[dst(e)], src(e))
            end
        end
    end
    return (unique!).((sort!).(old_neighbors)), (unique!).((sort!).(new_neighbors))
end

function _get_neighbors!(graph::HeapKNNGraph{V}, sample_rate, old_neighbors, new_neighbors) where V
    (empty!).(old_neighbors)
    (empty!).(new_neighbors)
    for e in edges(graph)
        if flag(e) # isnew(e) => new edges haven't participated in local join
            if rand() ≤ sample_rate 
                # mark sampled new forward neighbors as old
                e.flag = false
                push!(new_neighbors[src(e)], dst(e))
                push!(new_neighbors[dst(e)], src(e))
            end
        else # old neighbors
            # always include old forward
            push!(old_neighbors[src(e)], dst(e))
            # sample old reverse neighbors
            if rand() ≤ sample_rate
                push!(old_neighbors[dst(e)], src(e))
            end
        end
    end
    return (unique!).((sort!).(old_neighbors)), (unique!).((sort!).(new_neighbors))
end
