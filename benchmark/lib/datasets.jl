# Datasets for the benchmark suite.
#
# Two sources, matching the two tiers:
#
#  * Synthetic gaussian "blobs" (and uniform, as a stress case). Clustered data
#    has the low-intrinsic-dimensional structure NNDescent is meant to exploit
#    and is far more representative of real embeddings than uniform-in-a-cube
#    (which is close to the worst case for ANN). Float32, matching real ANN.
#
#  * Real ann-benchmarks datasets, loaded from the canonical HDF5 files at
#    ann-benchmarks.com. These ship train/test splits *and* exact ground-truth
#    neighbors, giving direct parity with the ann-benchmarks recall numbers.
#
# A `Dataset` bundles everything the recall harness needs. `train`/`test` are
# `Vector{Vector{Float32}}` (one entry per point) so they drop straight into
# `nndescent` and `search`.

struct Dataset
    name::String
    train::Vector{Vector{Float32}}
    test::Vector{Vector{Float32}}
    truth::Union{Matrix{Int},Nothing}   # k×ntest exact neighbors of test in train (1-based), or nothing
    metric::PreMetric
    metric_name::String
end

Base.show(io::IO, d::Dataset) = print(io,
    "Dataset($(d.name): $(length(d.train)) train × $(length(first(d.train)))d, ",
    "$(length(d.test)) test, metric=$(d.metric_name))")

# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

_cols(m::AbstractMatrix) = [Vector{Float32}(c) for c in eachcol(m)]

"""
    synthetic_blobs(n, d; n_clusters, n_test, metric, seed) -> Dataset

`n` points in `d` dimensions drawn from `n_clusters` isotropic gaussians with
random centers, plus `n_test` held-out queries from the same mixture. Ground
truth is computed by brute force (kept small via `n_test`).
"""
function synthetic_blobs(n::Integer, d::Integer;
                         n_clusters::Integer = 10,
                         n_test::Integer = 0,
                         k_truth::Integer = 100,
                         metric::PreMetric = Euclidean(),
                         metric_name::AbstractString = "euclidean",
                         seed::Integer = 0)
    rng = MersenneTwister(seed)
    centers = [10f0 .* randn(rng, Float32, d) for _ in 1:n_clusters]
    gen(m) = begin
        out = Vector{Vector{Float32}}(undef, m)
        for i in 1:m
            c = centers[rand(rng, 1:n_clusters)]
            out[i] = c .+ randn(rng, Float32, d)
        end
        out
    end
    train = gen(n)
    test = n_test > 0 ? gen(n_test) : Vector{Vector{Float32}}()
    truth = n_test > 0 ? bruteforce_knn(train, test, min(k_truth, n), metric) : nothing
    return Dataset("blobs-$(n)x$(d)", train, test, truth, metric, String(metric_name))
end

"""
    synthetic_uniform(n, d; ...) -> Dataset

Uniform in `[-1, 1]^d`. Near worst-case for ANN (no neighbor structure); kept as
an explicit stress case rather than a default.
"""
function synthetic_uniform(n::Integer, d::Integer;
                           n_test::Integer = 0,
                           k_truth::Integer = 100,
                           metric::PreMetric = Euclidean(),
                           metric_name::AbstractString = "euclidean",
                           seed::Integer = 0)
    rng = MersenneTwister(seed)
    gen(m) = [2f0 .* rand(rng, Float32, d) .- 1f0 for _ in 1:m]
    train = gen(n)
    test = n_test > 0 ? gen(n_test) : Vector{Vector{Float32}}()
    truth = n_test > 0 ? bruteforce_knn(train, test, min(k_truth, n), metric) : nothing
    return Dataset("uniform-$(n)x$(d)", train, test, truth, metric, String(metric_name))
end

# ---------------------------------------------------------------------------
# Real (ann-benchmarks HDF5)
# ---------------------------------------------------------------------------

const ANN_BENCH_BASE = "http://ann-benchmarks.com"

datasets_cache_dir() = get(ENV, "NND_BENCH_DATA", joinpath(@__DIR__, "..", ".datasets"))

function _download_dataset(name::AbstractString)
    dir = datasets_cache_dir()
    mkpath(dir)
    path = joinpath(dir, name * ".hdf5")
    if !isfile(path)
        url = "$(ANN_BENCH_BASE)/$(name).hdf5"
        @info "downloading dataset" name url dest=path
        Downloads.download(url, path)
    end
    return path
end

_metric_for(name::AbstractString) =
    occursin("angular", name) ? (CosineDist(), "cosine") :
    occursin("euclidean", name) ? (Euclidean(), "euclidean") :
    error("unknown metric for dataset $name")

"""
    load_ann_dataset(name; max_train, max_test) -> Dataset

Load a canonical ann-benchmarks dataset (e.g. `"fashion-mnist-784-euclidean"`),
downloading and caching the HDF5 file on first use. HDF5/numpy store points
row-major, so HDF5.jl reads the `train`/`test` arrays transposed (dims × points)
— exactly the column-of-points layout we want. The shipped `neighbors` array is
0-based; we shift to 1-based. `max_train`/`max_test` truncate for quicker runs
(this invalidates the shipped ground truth, so truth is recomputed when train is
truncated).
"""
function load_ann_dataset(name::AbstractString;
                          max_train::Integer = typemax(Int),
                          max_test::Integer = typemax(Int))
    HDF5 = Base.require(Base.PkgId(Base.UUID("f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"), "HDF5"))
    path = _download_dataset(name)
    metric, metric_name = _metric_for(name)
    train, test, neighbors = HDF5.h5open(path, "r") do f
        (Matrix{Float32}(read(f["train"])),
         Matrix{Float32}(read(f["test"])),
         Matrix{Int}(read(f["neighbors"])))
    end
    train_cols = _cols(train)
    test_cols = _cols(test)
    truncated_train = length(train_cols) > max_train
    if truncated_train
        train_cols = train_cols[1:max_train]
    end
    if length(test_cols) > max_test
        test_cols = test_cols[1:max_test]
        neighbors = neighbors[:, 1:max_test]
    end
    truth = if truncated_train
        # shipped neighbors index the full train set; recompute against the subset
        bruteforce_knn(train_cols, test_cols, min(100, length(train_cols)), metric)
    else
        neighbors .+ 1   # 0-based -> 1-based
    end
    return Dataset(name, train_cols, test_cols, truth, metric, metric_name)
end

"""
    load_dataset(spec; tier) -> Dataset

Resolve a dataset `spec` to a `Dataset`. A `spec` is either a real
ann-benchmarks name (`String`) or a synthetic descriptor
`(:blobs, n, d)` / `(:uniform, n, d)`.
"""
function load_dataset(spec; n_test::Integer = 1_000, n_clusters::Integer = 10, seed::Integer = 0)
    if spec isa AbstractString
        return load_ann_dataset(spec)
    elseif spec isa Tuple && spec[1] === :blobs
        return synthetic_blobs(spec[2], spec[3]; n_clusters=n_clusters, n_test=n_test, seed=seed)
    elseif spec isa Tuple && spec[1] === :uniform
        return synthetic_uniform(spec[2], spec[3]; n_test=n_test, seed=seed)
    else
        error("unrecognized dataset spec: $(repr(spec))")
    end
end
