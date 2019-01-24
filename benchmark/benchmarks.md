# Benchmark Report for *NearestNeighborDescent*

## Job Properties
* Time of benchmark: 24 Jan 2019 - 12:4
* Package commit: e39d08
* Julia commit: 80516c
* Julia command flags: None
* Environment variables: None

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                               | time            | GC time    | memory          | allocations |
|--------------------------------------------------|----------------:|-----------:|----------------:|------------:|
| `["graph", "random", "(:matrices, :cosine)"]`    | 583.662 ms (5%) |  61.617 ms | 275.22 MiB (1%) |     7605271 |
| `["graph", "random", "(:matrices, :euclidean)"]` | 834.666 ms (5%) | 115.472 ms | 505.38 MiB (1%) |    16973578 |
| `["graph", "random", "(:matrices, :hamming)"]`   | 430.898 ms (5%) |  57.861 ms | 267.02 MiB (1%) |     7152895 |
| `["graph", "random", "(:vectors, :cosine)"]`     | 585.303 ms (5%) |  60.070 ms | 274.31 MiB (1%) |     7568061 |
| `["graph", "random", "(:vectors, :euclidean)"]`  | 819.817 ms (5%) | 110.749 ms | 495.73 MiB (1%) |    16549078 |
| `["graph", "random", "(:vectors, :hamming)"]`    | 429.776 ms (5%) |  57.242 ms | 267.50 MiB (1%) |     7187191 |
| `["graph", "real", "fmnist"]`                    |   10.692 s (5%) |    2.869 s |   2.68 GiB (1%) |    81524692 |
| `["graph", "real", "mnist"]`                     |   18.923 s (5%) |    4.611 s |   4.91 GiB (1%) |   166020973 |
| `["query", "random", "(:matrices, :cosine)"]`    | 464.468 ms (5%) |  33.468 ms | 148.90 MiB (1%) |     5936839 |
| `["query", "random", "(:matrices, :euclidean)"]` | 363.391 ms (5%) |  32.219 ms | 148.84 MiB (1%) |     6176074 |
| `["query", "random", "(:matrices, :hamming)"]`   | 408.912 ms (5%) |  32.923 ms | 143.56 MiB (1%) |     5396341 |
| `["query", "random", "(:vectors, :cosine)"]`     | 492.783 ms (5%) |  32.667 ms | 149.41 MiB (1%) |     5958595 |
| `["query", "random", "(:vectors, :euclidean)"]`  | 368.710 ms (5%) |  29.688 ms | 148.83 MiB (1%) |     6174052 |
| `["query", "random", "(:vectors, :hamming)"]`    | 432.157 ms (5%) |  27.741 ms | 143.18 MiB (1%) |     5378329 |
| `["query", "real", "fmnist"]`                    |    2.806 s (5%) | 290.588 ms |   1.14 GiB (1%) |    25505761 |
| `["query", "real", "mnist"]`                     |    6.926 s (5%) | 729.643 ms |   2.13 GiB (1%) |    68754270 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["graph", "random"]`
- `["graph", "real"]`
- `["query", "random"]`
- `["query", "real"]`

## Julia versioninfo
```
Julia Version 1.1.0
Commit 80516ca202 (2019-01-21 21:24 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 18.04.1 LTS
  uname: Linux 4.15.0-36-generic #39-Ubuntu SMP Mon Sep 24 16:19:09 UTC 2018 x86_64 x86_64
  CPU: Intel(R) Core(TM) i5-7600K CPU @ 3.80GHz: 
              speed         user         nice          sys         idle          irq
       #1  4201 MHz    3269999 s      14176 s     747085 s  606279337 s          0 s
       #2  4200 MHz    2046126 s       9428 s    6294687 s  596582103 s          0 s
       #3  4200 MHz    3259799 s      12845 s     728007 s  606316575 s          0 s
       #4  4201 MHz    3530319 s      12908 s     688452 s  605939147 s          0 s
       
  Memory: 15.605533599853516 GB (8347.09765625 MB free)
  Uptime: 6.108036e6 sec
  Load Avg:  1.06689453125  0.81005859375  0.486328125
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, skylake)
```