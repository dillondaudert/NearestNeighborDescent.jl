
@testset "utilities tests" begin
    @testset "sample_neighbors" begin
        @testset "sample_rate = 1. tests" begin
            points = collect(1:10)
            n_neighbors = 4

            # zero neighbors
            idxs = sample_neighbors(length(points), 0)
            @test length(idxs) == 0

            # k < n
            idxs = sample_neighbors(length(points), n_neighbors)
            @test length(idxs) == n_neighbors
            @test issubset(idxs, points)

            # k > n
            idxs = sample_neighbors(length(points), length(points)+5)
            @test length(idxs) == length(points)
        end
        @testset "sample_rate = .5 tests" begin
            points = collect(1:10)
            n_neighbors = 4
            ρ = .5

            # zero neighbors
            idxs = sample_neighbors(length(points), 0, ρ)
            @test length(idxs) == 0

            # k < n
            idxs = sample_neighbors(length(points), n_neighbors, ρ)
            @test ρ*n_neighbors ≥ length(idxs)
            @test issubset(idxs, points)

            # k > n
            idxs = sample_neighbors(length(points), 2*length(points), ρ)
            @test length(idxs) == length(points)
        end
        @testset "exclude set tests" begin
            points = collect(1:10)

            # exclude 1
            idxs = sample_neighbors(length(points),
                                              length(points),
                                              exclude=[1])
            @test idxs ⊊ points
            @test !(1 ∈ idxs)

            # exclude all
            idxs = sample_neighbors(length(points),
                                              length(points),
                                              exclude=points)
            @test length(idxs) == 0

        end
    end

    #=
    @testset "neighbors tests" begin
        # create heaps
        knn_heaps = [BinaryMaxHeap{NNTuple{Int, Float64}}() for _ in 1:5]
        push!(knn_heaps[1], NNTuple(2, 100.))
        push!(knn_heaps[1], NNTuple(5, 100.))
        push!(knn_heaps[2], NNTuple(4, 100.))
        push!(knn_heaps[3], NNTuple(2, 100.))
        push!(knn_heaps[3], NNTuple(4, 100.))
        push!(knn_heaps[4], NNTuple(1, 100.))
        push!(knn_heaps[5], NNTuple(3, 100.))
        push!(knn_heaps[5], NNTuple(4, 100.))
        old_fw, fw, old_bw, bw = _neighbors(knn_heaps)

        @testset "new fw neighbors tests" begin
            @test fw[1] == [2, 5]
            @test fw[2] == [4]
            @test fw[3] == [2, 4]
            @test fw[4] == [1]
            @test fw[5] == [3, 4]
            @test all(isempty(v) for v in old_fw)
        end
        @testset "new bw neighbors tests" begin
            @test bw[1] == [4]
            @test bw[2] == [1, 3]
            @test bw[3] == [5]
            @test bw[4] == [2, 3, 5]
            @test bw[5] == [1]
            @test all(isempty(v) for v in old_bw)
        end
        old_fw, fw, old_bw, bw = _neighbors(knn_heaps)
        @testset "old fw neighbors tests" begin
            @test old_fw[1] == [2, 5]
            @test old_fw[2] == [4]
            @test old_fw[3] == [2, 4]
            @test old_fw[4] == [1]
            @test old_fw[5] == [3, 4]
            @test all(isempty(v) for v in fw)
        end
        @testset "old bw neighbors tests" begin
            @test old_bw[1] == [4]
            @test old_bw[2] == [1, 3]
            @test old_bw[3] == [5]
            @test old_bw[4] == [2, 3, 5]
            @test old_bw[5] == [1]
            @test all(isempty(v) for v in bw)
        end
    end
    =#

    @testset "_heappush! tests" begin
        @testset "max_length tests" begin
            h = BinaryMaxHeap{HeapKNNGraphEdge{Int, Float64}}()
            e = HeapKNNGraphEdge(1, 2, 1.)
            KNNGraphs._heappush!(h, e, 0)
            @test length(h) == 0

            KNNGraphs._heappush!(h, e, 1)
            @test length(h) == 1
            @test top(h) == e

            e2 = HeapKNNGraphEdge(1, 3, 1.) # new point, heappush! just replaces
            KNNGraphs._heappush!(h, e2, 1)
            @test length(h) == 1
            @test top(h) == e2
        end
    end
end # end utilities tests
