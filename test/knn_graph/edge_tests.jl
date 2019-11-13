
@testset "HeapKNNGraphEdge tests" begin

    @testset "constructor" begin
        @test_throws ErrorException HeapKNNGraphEdge(1, 1, rand())
    end
    @testset "interface" begin
        @test_broken false
    end
end
