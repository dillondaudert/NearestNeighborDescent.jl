# Public Interface
Documentation for `NearestNeighborDescent.jl`'s public interface.

## Contents
```@contents
Pages = ["public.md"]
```

## Index
```@index
Pages = ["public.md"]
```

## Public Interface

```@docs
nndescent
search
local_join!
get_neighbors!
```

### KNN Graphs Public Interface

```@docs
KNNGraphs
ApproximateKNNGraph
HeapKNNGraph
LockHeapKNNGraph
knn_diameter
knn_matrices
flag
weight
edge_indices
node_edge
node_edges
update_flag!
```
