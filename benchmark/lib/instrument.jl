# Distance-evaluation counting.
#
# The hardware-invariant cost metric for NNDescent is the number of distance
# computations ("scan rate" in Dong et al. 2011, = #distance evals / n^2). Wall
# time conflates this with cache behaviour, allocation, and threading; counting
# evaluations directly lets the recall harness report the algorithm's intrinsic
# work so regressions in *algorithmic* efficiency are visible separately from
# constant-factor / hardware effects.
#
# `CountingMetric` wraps a `SemiMetric` (Euclidean, SqEuclidean, CosineDist, ...)
# and tallies every `evaluate` call. It is declared `<: SemiMetric` so the
# symmetry fast-path in `local_join!` is taken exactly as for the wrapped metric.
# Counting is atomic so it stays correct under threaded construction, but because
# the atomic add perturbs timing it is meant for the (untimed) scan-rate pass.

struct CountingMetric{M<:SemiMetric} <: SemiMetric
    metric::M
    count::Threads.Atomic{Int}
end

CountingMetric(m::SemiMetric) = CountingMetric(m, Threads.Atomic{Int}(0))

@inline function Distances.evaluate(d::CountingMetric, a, b)
    Threads.atomic_add!(d.count, 1)
    return Distances.evaluate(d.metric, a, b)
end

# Distances dispatches both as `evaluate(metric, a, b)` and as a functor.
@inline (d::CountingMetric)(a, b) = Distances.evaluate(d, a, b)

Distances.result_type(d::CountingMetric, a, b) = Distances.result_type(d.metric, a, b)

count_evals(d::CountingMetric) = d.count[]
reset_count!(d::CountingMetric) = (d.count[] = 0; d)

"""
    scan_rate(n, evals) -> Float64

Distance evaluations as a fraction of the brute-force `n^2`. This is the
scale-invariant quantity NNDescent claims grows like `n^0.14` (i.e. scan rate
shrinks as `n` grows); tracking it across the size sweep validates that claim.
"""
scan_rate(n::Integer, evals::Integer) = evals / (float(n) * n)
