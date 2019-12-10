
@testset "search tests" begin

    @testset "basic usage tests" begin
        for GraphT in [HeapKNNGraph, LockHeapKNNGraph]
            data = [rand(5) for _ in 1:20]
            queries = [rand(5) for _ in 1:5]
            graph = nndescent(GraphT, data, 4, Euclidean(); max_iters=1)
            # vector of vector usage
            @inferred search(graph, queries, 3)

            data = rand(5, 20)
            queries = rand(5, 5)
            # matrix usage
            @inferred search(graph, queries, 3)
        end
        @test true
    end

    Random.seed!(0)
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
    true_inds = [1 3;
                 2 4]
    true_dists = [.4 .4;
                  .6 .6]

    for GraphT in [HeapKNNGraph, LockHeapKNNGraph]
        graph = nndescent(GraphT, data, 3, Euclidean())
        inds, dists = search(graph, queries, 2)

        @test all(inds .== true_inds)
        @test all(dists .== true_dists)
    end

end
