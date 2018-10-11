
@testset "sample neighbors tests" begin
    @testset "sample_rate = 1. tests" begin
        points = collect(1:10)
        n_neighbors = 4

        # zero neighbors
        idxs = NNDescent.sample_neighbors(length(points), 0)
        @test length(idxs) == 0

        # k < n
        idxs = NNDescent.sample_neighbors(length(points), n_neighbors)
        @test length(idxs) == n_neighbors
        @test issubset(idxs, points)

        # k > n
        idxs = NNDescent.sample_neighbors(length(points), length(points)+5)
        @test length(idxs) == length(points)
    end
    @testset "sample_rate = .5 tests" begin
        points = collect(1:10)
        n_neighbors = 4
        ρ = .5

        # zero neighbors
        idxs = NNDescent.sample_neighbors(length(points), 0, ρ)
        @test length(idxs) == 0

        # k < n
        idxs = NNDescent.sample_neighbors(length(points), n_neighbors, ρ)
        @test ρ*n_neighbors ≥ length(idxs)
        @test issubset(idxs, points)

        # k > n
        idxs = NNDescent.sample_neighbors(length(points), 2*length(points), ρ)
        @test length(idxs) == length(points)
    end
    @testset "exclude set tests" begin
        points = collect(1:10)

        # exclude 1
        idxs = NNDescent.sample_neighbors(length(points),
                                          length(points),
                                          exclude=[1])
        @test idxs ⊊ points
        @test !(1 ∈ idxs)

        # exclude all
        idxs = NNDescent.sample_neighbors(length(points),
                                          length(points),
                                          exclude=points)
        @test length(idxs) == 0

    end
end

@testset "_init_knn_tree tests" begin
    data = Vector([rand(3) for _ in 1:10])
    n_neighbors = 3

    knn_tree = NNDescent._init_knn_tree(data, n_neighbors)

    @test length(knn_tree) == length(data)
    for p in 1:length(knn_tree)
        @test length(knn_tree[p]) == n_neighbors
        while !isempty(knn_tree[p])
            t = pop!(knn_tree[p])
            @test t.idx ≤ length(data)
            @test t.dist == Inf
        end
    end
end
