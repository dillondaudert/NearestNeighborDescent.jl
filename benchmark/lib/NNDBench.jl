"""
    NNDBench

Support library for the NearestNeighborDescent.jl benchmark suite. Provides the
pieces a credible ANN benchmark needs beyond raw timing:

  * tiered workload configuration (`tier_config`, `current_tier`)
  * realistic + real datasets with ground truth (`Dataset`, `load_dataset`,
    `synthetic_blobs`, `load_ann_dataset`)
  * exact KNN and recall scoring (`bruteforce_knn`, `recall`)
  * distance-evaluation counting / scan rate (`CountingMetric`, `scan_rate`)

Used by the timing suite (`benchmarks.jl`) and the recall and thread-scaling
harnesses under `benchmark/harness/`.
"""
module NNDBench

using Random
using Statistics
using Distances
using Downloads
using NearestNeighborDescent

include("config.jl")
include("instrument.jl")
include("groundtruth.jl")
include("datasets.jl")

export tier_config, current_tier
export Dataset, load_dataset, load_ann_dataset, synthetic_blobs, synthetic_uniform
export bruteforce_knn, bruteforce_self_knn, recall
export CountingMetric, count_evals, reset_count!, scan_rate

end # module
