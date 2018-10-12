
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

@testset "neighbors tests" begin
    # create tree
    knn_tree = [mutable_binary_maxheap(_NNTuple{Int, Float64}) for _ in 1:5]
    # add some neighbors
    push!(knn_tree[1], _NNTuple(2, Inf))
    push!(knn_tree[1], _NNTuple(5, Inf))
    push!(knn_tree[2], _NNTuple(4, Inf))
    push!(knn_tree[3], _NNTuple(2, Inf))
    push!(knn_tree[3], _NNTuple(4, Inf))
    push!(knn_tree[4], _NNTuple(1, Inf))
    push!(knn_tree[5], _NNTuple(3, Inf))
    push!(knn_tree[5], _NNTuple(4, Inf))

    @testset "fw neighbors tests" begin
        fw = _fw_neighbors(knn_tree)
        @test fw[1] == [2, 5]
        @test fw[2] == [4]
        @test fw[3] == [2, 4]
        @test fw[4] == [1]
        @test fw[5] == [3, 4]
    end
    @testset "bw neighbors tests" begin
        bw = _bw_neighbors(knn_tree)
        @test bw[1] == [4]
        @test bw[2] == [1, 3]
        @test bw[3] == [5]
        @test bw[4] == [2, 3, 5]
        @test bw[5] == [1]
    end
end

@testset "_update_nn tests" begin
    knn_tree = [mutable_binary_maxheap(_NNTuple{Int, Float64})]
    push!(knn_tree[1], _NNTuple(2, 10.))
    push!(knn_tree[1], _NNTuple(5, 20.))
    _update_nn(knn_tree, 1, _NNTuple(3, 8.))
    @test top(knn_tree[1]) == _NNTuple(2, 10.)
end
