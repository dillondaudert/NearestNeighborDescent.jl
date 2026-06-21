# Release History

## v0.4.1

Bug-fix release addressing a `search` crash on realistic inputs and incorrect
thread-local buffer handling, plus a threaded-construction speedup.

### Bug Fixes

- **`search` no longer errors on larger datasets**: the bounded candidate queue
  compared a scalar distance against a whole `(distance, index, flag)` heap tuple,
  throwing `MethodError: isless(::Float64, ::Tuple)`. This only triggered once an
  unvisited neighbor was reached — i.e. whenever `max_candidates < nv(graph)`, the
  normal case for real data — so the existing tests never hit it.
- **Thread-safe `seen` buffers in `search`**: replaced the `@threads :static` loop
  that indexed a per-thread buffer by `Threads.threadid()` with a `Channel`-based
  buffer pool. The previous pattern could error or corrupt results under task
  migration and on Julia 1.12, where the interactive thread occupies thread id 1
  and worker thread ids start at 2 (see
  [PSA: don't use threadid()](https://julialang.org/blog/2023/07/PSA-dont-use-threadid/)).

### Performance

- **Faster threaded `LockHeapKNNGraph` construction**: a captured variable that was
  reassigned inside the threaded init loop got boxed as `Any`, forcing runtime
  dispatch on every edge insertion. Splitting it into single-assignment bindings
  restores type inference (~2.5× faster construction, ~76% fewer allocations on the
  benchmark workload).

### Internal Changes

- **CI runs the test suite multithreaded** (`JULIA_NUM_THREADS=2,1`) so the threaded
  code paths and the Julia 1.12 interactive-thread layout are actually exercised.
- **JET type-stability checks**: a dedicated CI step runs `@report_call`/`@report_opt`
  over the `nndescent` and `search` entry points (`test/jet_tests.jl`) to catch latent
  method errors and runtime dispatch that the behavioral tests miss.

## v0.4.0

Maintenance release that raises the minimum Julia version and modernizes the
package's CI, dependency management, and documentation tooling.

### Breaking Changes

- **Minimum Julia version is now 1.10 (LTS)**: support for Julia 1.5–1.9 has been
  dropped. The `julia` compat bound is now `1.10, 1.11, 1.12`.

### Internal Changes

- **Pkg workspace**: the root `Project.toml` now declares a `[workspace]` covering
  the `test/`, `docs/`, and `benchmark/` sub-projects, so they resolve against a
  single shared manifest.
- **DataStructures deprecation removed**: `search` now uses `first(heap)` instead of
  the deprecated `top(heap)` for inspecting the candidate heap.
- **Dependency updates via Dependabot**: replaced the CompatHelper workflow with
  GitHub-native Dependabot (`.github/dependabot.yml`), which also keeps GitHub
  Actions versions current ([PSA](https://discourse.julialang.org/t/psa-github-dependabot-now-supports-julia/134997)).
- **Documentation modernized to Documenter 1.x**: bumped `Documenter` from `0.24` to
  `1`, simplified `docs/make.jl`, and moved the doc build into a dedicated
  `documentation.yml` workflow (also run on pull requests).
- **CI**: the test matrix now covers `min`/`lts`/`1` Julia versions, and coverage is
  uploaded to both Codecov and Coveralls via Coverage.jl.

## v0.3.x and earlier

For release notes from v0.3.x releases and earlier, see the
[GitHub releases page](https://github.com/dillondaudert/NearestNeighborDescent.jl/releases).
