
@testset "DescentGraph tests" begin
    @testset "Vector{Vector} constructor tests" begin
        data = [[0., 0.],
                [0., 1.]]
        graph = DescentGraph(data, 1)
        @test size(graph.indices) == (1, 2)
        @test size(graph.distances) == (1, 2)
        @test graph.indices == [2 1]
        @test graph.distances == [1. 1.]

        for np = 20:10:50, k = 1:2:10
            data = [rand(5) for _ in 1:np]
            graph = DescentGraph(data, k)
            @test size(graph.indices) == (k, np)
            @test size(graph.distances) == (k, np)
        end
    end
    @testset "Matrix constructor tests" begin
        data = [0. 0.; 0. 1.]
        @test DescentGraph(data, 1) isa DescentGraph
    end
    @testset "eltype stability tests" begin
        # Float32
        data = [rand(Float32, 10) for _ in 1:100]
        @inferred DescentGraph(data, 5)
        # Integer
        data = [rand([0, 1], 10) for _ in 1:100]
        @inferred DescentGraph(data, 5, Hamming())
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
            @test sort(graph.indices, dims=1) == sort(true_3nn, dims=1)
        end
    end
    @testset "CosineDist" begin
        data = [rand(50) for _ in 1:10]
        true_3nn = brute_knn(data, CosineDist(), 3)
        desc_3nn = DescentGraph(data, 3, CosineDist()).indices
        @test_skip getindex.(true_3nn, 1) == desc_3nn
    end
    @testset "Hamming" begin
        data = [rand([0, 1], 50) for _ in 1:10]
        true_3nn = brute_knn(data, Hamming(), 3)
        desc_3nn = DescentGraph(data, 3, Hamming()).indices
        @test_skip getindex.(true_3nn, 1) == desc_3nn
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
    
    @testset "search matrix queries tests" begin
        queries = [0. 0.; 0. 1.; 0.4 0.4]
        inds, dists = search(graph, queries, 2, 4)
        @test all(inds .== true_inds)
        @test all(dists .== true_dists)
    end
    
    
end
