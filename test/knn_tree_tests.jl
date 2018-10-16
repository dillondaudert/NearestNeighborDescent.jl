
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
        for t = 1:length(knn_tree[p])
            @test knn_tree[p][t].idx ≠ p
            @test knn_tree[p][t].idx ≤ length(data)
            @test knn_tree[p][t].dist == Inf
        end
    end
end

@testset "neighbors tests" begin
    # create tree
    knn_tree = [[NNTuple(2, Inf), NNTuple(5, Inf)],
                [NNTuple(4, Inf)],
                [NNTuple(2, Inf), NNTuple(4, Inf)],
                [NNTuple(1, Inf)],
                [NNTuple(3, Inf), NNTuple(4, Inf)]]
    old_fw, fw, old_bw, bw = _neighbors(knn_tree)

    @testset "new fw neighbors tests" begin
        @test fw[1] == [2, 5]
        @test fw[2] == [4]
        @test fw[3] == [2, 4]
        @test fw[4] == [1]
        @test fw[5] == [3, 4]
    end
    @testset "new bw neighbors tests" begin
        @test bw[1] == [4]
        @test bw[2] == [1, 3]
        @test bw[3] == [5]
        @test bw[4] == [2, 3, 5]
        @test bw[5] == [1]
    end
    old_fw, fw, old_bw, bw = _neighbors(knn_tree)
    @testset "old fw neighbors tests" begin
        @test old_fw[1] == [2, 5]
        @test old_fw[2] == [4]
        @test old_fw[3] == [2, 4]
        @test old_fw[4] == [1]
        @test old_fw[5] == [3, 4]
    end
    @testset "old bw neighbors tests" begin
        @test old_bw[1] == [4]
        @test old_bw[2] == [1, 3]
        @test old_bw[3] == [5]
        @test old_bw[4] == [2, 3, 5]
        @test old_bw[5] == [1]
    end
end

@testset "_update_nn tests" begin

    @testset "no changes tests" begin
        v_knn = [NNTuple(1, 10.), NNTuple(2, 20.), NNTuple(3, 30.)]
        @test _update_nn!(v_knn, NNTuple(4, 40.)) == 0
        @test length(v_knn) == 3
        @test v_knn[end].idx == 3
        @test v_knn[end].dist == 30.
    end
    @testset "exists tests" begin
        v_knn = [NNTuple(1, 10.), NNTuple(2, 20.), NNTuple(3, Inf)]
        @test _update_nn!(v_knn, NNTuple(3, 5.)) == 1
        @test v_knn[1].idx == 3
        @test v_knn[1].dist == 5.
        @test v_knn[end].idx == 2
        @test v_knn[end].dist == 20.
        @test _update_nn!(v_knn, NNTuple(3, 5.)) == 0
    end
    @testset "new nearest neighbor tests" begin
        v_knn = [NNTuple(1, 10.), NNTuple(2, 20.), NNTuple(3, 30.)]
        @test _update_nn!(v_knn, NNTuple(4, 15.)) == 1
        @test length(v_knn) == 3
        @test v_knn[2].idx == 4
        @test v_knn[2].dist == 15.
        @test v_knn[end].idx == 2
        @test v_knn[1].idx == 1
    end
end
