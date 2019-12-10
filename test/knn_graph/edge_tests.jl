
@testset "HeapKNNGraphEdge tests" begin

    @testset "constructor" begin
        @test_throws ErrorException HeapKNNGraphEdge(1, 1, rand())
        @test_throws ErrorException HeapKNNGraphEdge(2, 2, rand(), true)
        @test HeapKNNGraphEdge(1, 2, rand()) isa HeapKNNGraphEdge{Int, Float64}
        @test HeapKNNGraphEdge(2, 1, rand(), false) isa HeapKNNGraphEdge
    end
    @testset "interface" begin
        @testset "comparison" begin
            e1 = HeapKNNGraphEdge(1, 2, .5, true)
            e2 = HeapKNNGraphEdge(1, 2, .5, true)
            e3 = HeapKNNGraphEdge(1, 2, .5, false)
            @test e1 == e2
            @test e1 === e2
            @test e1 == e3 # `flag` does not participate in normal equality
            @test e1 !== e3 # but does for `===`
            # floating point equality has a tolerance
            e4 = HeapKNNGraphEdge(1, 2, .5 + .25*sqrt(eps(Float64)), false)
            @test e3 == e4 # within tol
            @test !(e3 < e4)
            e5 = HeapKNNGraphEdge(1, 2, .5 + sqrt(eps(Float64)), false)
            @test e3 != e5 # outside of tol
            @test e3 < e5
            # rational weights
            e1 = HeapKNNGraphEdge(1, 2, 1//3, true)
            e2 = HeapKNNGraphEdge(1, 2, 1//3, true)
            e3 = HeapKNNGraphEdge(1, 2, 1//2, true)
            @test e1 == e2
            @test e1 === e2
            @test e1 != e3
            @test e1 < e3
        end
        e1 = HeapKNNGraphEdge(1, 2, .5, true)
        @test eltype(e1) == Int

        @testset "property methods" begin
            e1 = HeapKNNGraphEdge(3, 4, 0., true)
            @test src(e1) == 3
            @test dst(e1) == 4
            @test weight(e1) == 0.
            @test flag(e1)
            rev_e1 = reverse(e1)
            e2 = HeapKNNGraphEdge(4, 3, 0., true)
            @test rev_e1 === e2
        end

    end
end
