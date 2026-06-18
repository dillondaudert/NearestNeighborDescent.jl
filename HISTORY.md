# Release History

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
