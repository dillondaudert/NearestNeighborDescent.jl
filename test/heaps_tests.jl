# tests for min-max heaps

@testset "minmax heaps tests" begin

    @testset "is_minmax_heap tests" begin
        mmheap = [0, 10, 9, 2, 3, 4, 5]
        @test is_minmax_heap(mmheap)
        @test is_minmax_heap([])
        @test is_minmax_heap([rand()])
    end

    @testset "minmax_heapify tests" begin
        for i = 1:20
            @test is_minmax_heap(minmax_heapify!(rand(rand(0:200))))
        end
    end

end
