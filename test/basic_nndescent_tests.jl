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
        
    end
end
