using PkgBenchmark
benchmarkpkg(
    dirname(@__DIR__),
    BenchmarkConfig(
        env = Dict(
            "JULIA_NUM_THREADS" => "4",
        ),
    ),
    resultfile = joinpath(@__DIR__, "result.json")
)
