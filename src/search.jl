

"""
    search(graph, queries, n_neighbors; max_candidates) -> indices, distances

Search the kNN `graph` for the nearest neighbors of the points in `queries`.
`max_candidates` controls how large the candidate queue should be (min `n_neighbors`);
larger values increase accuracy at the cost of speed.
"""
function search end
