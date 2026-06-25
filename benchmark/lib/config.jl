# Benchmark tiers.
#
# A "tier" fixes the size of the workloads so the same harness can run either as
# a fast, network-free CI regression check (`small`) or as a faithful,
# ann-benchmarks-scale evaluation (`heavy`). Select with the `NND_BENCH_TIER`
# environment variable; defaults to `small`.
#
# - `small`  : synthetic data only, fits in seconds, deterministic, no network.
#              Used by the PkgBenchmark timing suite and by CI.
# - `heavy`  : real ann-benchmarks datasets (downloaded HDF5) plus large
#              synthetic sizes. Minutes-to-hours; opt-in.

const DEFAULT_TIER = "small"

current_tier() = lowercase(get(ENV, "NND_BENCH_TIER", DEFAULT_TIER))

"""
    tier_config([tier]) -> NamedTuple

Return the workload parameters for `tier` (defaults to the `NND_BENCH_TIER`
environment variable). Fields:

- `synthetic_sizes` : Vector of `(n, d)` synthetic dataset sizes for the timing
   and thread-scaling suites.
- `n_clusters`      : number of gaussian clusters for synthetic "blob" data.
- `ks`              : neighbor counts `k` to sweep.
- `metrics`         : `(name, metric)` pairs to sweep in the timing suite.
- `real_datasets`   : ann-benchmarks dataset names for the recall harness.
- `thread_counts`   : thread counts for the scaling driver (clamped to ncpu).
- `query_count`     : number of queries for synthetic search benchmarks.
"""
function tier_config(tier::AbstractString = current_tier())
    if tier == "heavy"
        return (
            synthetic_sizes = [(10_000, 50), (50_000, 100), (100_000, 128)],
            n_clusters      = 50,
            ks              = [10, 30],
            metrics         = [("euclidean", Euclidean()), ("cosine", CosineDist())],
            real_datasets   = ["fashion-mnist-784-euclidean", "sift-128-euclidean", "glove-100-angular"],
            thread_counts   = [1, 2, 4, 8, 16],
            query_count     = 10_000,
        )
    elseif tier == "small"
        return (
            synthetic_sizes = [(2_000, 25), (10_000, 50)],
            n_clusters      = 10,
            ks              = [10, 20],
            metrics         = [("euclidean", Euclidean()), ("cosine", CosineDist())],
            real_datasets   = ["fashion-mnist-784-euclidean"],
            thread_counts   = [1, 2, 4],
            query_count     = 1_000,
        )
    else
        error("unknown benchmark tier $(repr(tier)); set NND_BENCH_TIER to \"small\" or \"heavy\"")
    end
end
