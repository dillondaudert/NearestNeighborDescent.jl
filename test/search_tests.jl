
@testset "search tests" begin
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

    graph = nndescent(data, 3, Euclidean())
    inds, dists = search(graph, data, queries, 2, Euclidean())

    @test all(inds .== true_inds)
    @test all(dists .== true_dists)

end
