

@testset "make_knn_heaps tests" begin
    @testset "Float64 tests" begin
        data = [rand(3) for _ in 1:10]
        n_neighbors = 3

        knn_heaps = make_knn_heaps(data, n_neighbors, Euclidean())

        @test length(knn_heaps) == length(data)
        for p in 1:length(knn_heaps)
            @test length(knn_heaps[p]) == n_neighbors
            for t = 1:length(knn_heaps[p])
                @test knn_heaps[p].valtree[t].idx ≠ p
                @test knn_heaps[p].valtree[t].idx ≤ length(data)
                d = evaluate(Euclidean(), data[p], data[knn_heaps[p].valtree[t].idx])
                @test knn_heaps[p].valtree[t].dist == d
            end
        end
    end
    @testset "Int tests" begin
        data = [rand([0, 1], 3) for _ in 1:10]
        n_neighbors = 2
        knn_heaps = make_knn_heaps(data, n_neighbors, Hamming())

        @test length(knn_heaps) == length(data)
        for p in 1:length(knn_heaps)
            @test length(knn_heaps[p]) == n_neighbors
            for t = 1:length(knn_heaps[p])
                @test knn_heaps[p].valtree[t].idx ≠ p
                @test knn_heaps[p].valtree[t].idx ≤ length(data)
                d = evaluate(Hamming(), data[p], data[knn_heaps[p].valtree[t].idx])
                @test knn_heaps[p].valtree[t].dist == d
            end
        end

    end
end

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

@testset "_heappush! tests" begin
    @testset "max_cand tests" begin
        h = BinaryMaxHeap{NNTuple{Int, Float64}}()
        t = NNTuple(1, 1.)
        _heappush!(h, t, 0)
        @test length(h) == 0

        _heappush!(h, t, 1)
        @test length(h) == 1
        @test top(h) == t

        d = NNTuple(2, .5)
        _heappush!(h, d, 1)
        @test length(h) == 1
        @test top(h) == d
    end

    @testset "return val tests" begin
        h = BinaryMaxHeap{NNTuple{Int, Float64}}()
        # max_cand
        @test _heappush!(h, NNTuple(1, rand()), 0) == 0
        # empty heap push
        @test _heappush!(h, NNTuple(1, 1.), 1) == 1
        # length == max AND further away, no push
        @test _heappush!(h, NNTuple(2, 2.), 1) == 0
        # length == max BUT closer, push
        @test _heappush!(h, NNTuple(3, .5), 1) == 1
        @test top(h).idx == 3
        @test top(h).dist == .5
        @test length(h) == 1
        # length < max AND further, push
        @test _heappush!(h, NNTuple(4, 4.), 2) == 1
        @test top(h).idx == 4
        @test top(h).dist == 4.
        # tuple already in heap, no push
        @test _heappush!(h, NNTuple(3, .5), 3) == 0
        @test length(h) == 2
        @test top(h).idx == 4
        @test top(h).dist == 4.
    end
end
