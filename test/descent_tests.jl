
@testset "descent tests" begin
    @testset "basic usage tests" begin
        data = [rand(5) for _ in 1:20]
        k = 4
        metric = SqEuclidean()
        # vector of vectors
        nndescent(data, k, metric; max_iters=1)

        # matrix
        data = rand(5, 20)
        nndescent(data, k, metric; max_iters=1)

        @test true

    end
    test_inds = Int[2 4 2 1 1;
                    5 3 4 3 4]
    test_dsts = Float64[1. 2. 3. 4. 5.;
                        2. 3. 4. 5. 6.]
    @testset "get_neighbors! tests" begin
        g = HeapKNNGraph(test_inds, test_dsts)
        olds, news = NearestNeighborDescent.get_neighbors!(g)
        @test all(length.(olds) .== 0) # no old edges
        @test news[1] == [2, 4, 5]
        @test news[2] == [1, 3, 4]
        @test news[3] == [2, 4]
        @test news[4] == [1, 2, 3, 5]
        @test news[5] == [1, 4]
        olds, news = NearestNeighborDescent.get_neighbors!(g)
        @test all(length.(news) .== 0)
        @test olds[1] == [2, 4, 5]
        @test olds[2] == [1, 3, 4]
        @test olds[3] == [2, 4]
        @test olds[4] == [1, 2, 3, 5]
        @test olds[5] == [1, 4]
    end

    cube_points = [[0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.]]
    cube_true_nn = transpose([2 3 5;
                         1 4 6;
                         1 4 7;
                         2 3 8;
                         1 6 7;
                         2 5 8;
                         3 5 8;
                         4 6 7])

    @testset "local_join tests" begin
        mean_knn_diameter(g) = mean([knn_diameter(g, v) for v in vertices(g)])
        Random.seed!(0)
        data = [rand(5) for _ in 1:8]
        g1 = HeapKNNGraph(data, 3, Euclidean())
        init_inds, init_dsts = knn_matrices(g1)
        g2 = LockHeapKNNGraph(init_inds, init_dsts)
        g1_diam = mean_knn_diameter(g1)

        NearestNeighborDescent.local_join!(g1, data, Euclidean())
        NearestNeighborDescent.local_join!(g2, data, Euclidean())
        @test mean_knn_diameter(g1) ≤ g1_diam # test the knn diameter is nonincreasing
        inds1, dsts1 = knn_matrices(g1)
        inds2, dsts2 = knn_matrices(g2)
        @test all(inds1 .== inds2)
        @test all(dsts1 .== dsts2)

    end

end
