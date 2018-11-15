using Distances: Euclidean

@testset "knn tests" begin
    data = [[0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.]]
    _3nn = brute_knn(data, Euclidean(), 3)
    true_3nn = [2 3 5;
                1 4 6;
                1 4 7;
                2 3 8;
                1 6 7;
                2 5 8;
                3 5 8;
                4 6 7]
    @testset "brute knn test" begin
        @test _3nn == true_3nn
    end
    @testset "basic_nndescent tests" begin
        tree = DescentTree(data, 3)
        ids, dists = knn(tree)
        descent_3nn = transpose(ids)
        @test sort(descent_3nn, dims=2) == sort(true_3nn, dims=2)
    end
end

@testset "heappush! tests" begin
    @testset "max_cand tests" begin
        h = binary_maxheap(NNTuple{Int, Float64})
        t = NNTuple(1, 1.)
        heappush!(h, t, 0)
        @test length(h) == 0

        heappush!(h, t, 1)
        @test length(h) == 1
        @test top(h) == t

        d = NNTuple(2, .5)
        heappush!(h, d, 1)
        @test length(h) == 1
        @test top(h) == d
    end

    @testset "return val tests" begin
        h = binary_maxheap(NNTuple{Int, Float64})
        # max_cand
        @test heappush!(h, NNTuple(1, rand()), 0) == 0
        # empty heap push
        @test heappush!(h, NNTuple(1, 1.), 1) == 1
        # length == max AND further away, no push
        @test heappush!(h, NNTuple(2, 2.), 1) == 0
        # length == max BUT closer, push
        @test heappush!(h, NNTuple(3, .5), 1) == 1
        @test top(h).idx == 3
        @test top(h).dist == .5
        @test length(h) == 1
        # length < max AND further, push
        @test heappush!(h, NNTuple(4, 4.), 2) == 1
        @test top(h).idx == 4
        @test top(h).dist == 4.
        # tuple already in heap, no push
        @test heappush!(h, NNTuple(3, .5), 3) == 0
        @test length(h) == 2
    end
end

@testset "search tests" begin
    data = [[0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.]]
    queries = [[0., 0., 0.4],
               [0., 1., 0.4]]

    tree = DescentTree(data, 3)
    cands = search(tree, queries, 2, 4)
    @test sort(cands[1].valtree)[1].idx == 1
    @test sort(cands[2].valtree)[1].idx == 3
end
