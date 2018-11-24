
@testset "DescentGraph tests" begin
    @testset "Constructor tests" begin
        data = [[0., 0.],
                [0., 1.]]
        graph = DescentGraph(data, 1)
        @test size(graph.graph) == (1, 2)
        @test graph.graph[1,1] == (2, 1.)
        @test graph.graph[1,2] == (1, 1.)

        for np = 20:10:50, k = 1:2:10
            data = [rand(5) for _ in 1:np]
            graph = DescentGraph(data, k)
            @test size(graph.graph) == (k, np)
        end
    end
end

@testset "Distances tests" begin
    @testset "Euclidean" begin
        data = [[0., 0., 0.],
                [0., 0., 1.],
                [0., 1., 0.],
                [0., 1., 1.],
                [1., 0., 0.],
                [1., 0., 1.],
                [1., 1., 0.],
                [1., 1., 1.]]
        _3nn = brute_knn(data, Euclidean(), 3)
        true_3nn = transpose([2 3 5;
                    1 4 6;
                    1 4 7;
                    2 3 8;
                    1 6 7;
                    2 5 8;
                    3 5 8;
                    4 6 7])
        @testset "brute knn test" begin
            @test sort(getindex.(_3nn, 1), dims=1) == sort(true_3nn, dims=1)
        end
        @testset "basic_nndescent tests" begin
            graph = DescentGraph(data, 3)
            @test sort(getindex.(graph.graph, 1), dims=1) == sort(true_3nn, dims=1)
        end
    end
    @testset "CosineDist" begin
        data = [rand(50) for _ in 1:10]
        _3nn = brute_knn(data, CosineDist(), 3)
        desc_3nn = DescentGraph(data, 3, CosineDist()).graph
        @test getindex.(_3nn, 1) == getindex.(desc_3nn, 1)
    end
    @testset "Hamming" begin
        data = [rand([0, 1], 50) for _ in 1:10]
        _3nn = brute_knn(data, Hamming(), 3)
        desc_3nn = DescentGraph(data, 3, Hamming()).graph
        @test_skip getindex.(_3nn, 1) == getindex.(desc_3nn, 1)
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

    graph = DescentGraph(data, 3)
    true_inds = [1 3;
                 2 4]
    true_dists = [.4 .4;
                  .6 .6]
    inds, dists = search(graph, queries, 2, 4)
    @test all(inds .== true_inds)
    @test all(dists .== true_dists)
end
