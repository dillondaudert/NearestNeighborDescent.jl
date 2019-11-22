
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
