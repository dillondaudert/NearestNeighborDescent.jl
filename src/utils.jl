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
