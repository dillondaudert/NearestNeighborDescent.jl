# tests for min-max heaps
using NearestNeighborDescent: is_minmax_heap, minmax_heapify!, heapmin, heapmax

@testset "minmax heaps tests" begin

    @testset "is_minmax_heap tests" begin
        mmheap = [0, 10, 9, 2, 3, 4, 5]
        @test is_minmax_heap(mmheap)
        @test is_minmax_heap([])
        @test is_minmax_heap([rand()])
    end

    @testset "minmax_heapify tests" begin
        for i = 1:20
            A = rand(50)
            minmax_heapify!(A)
            @test is_minmax_heap(A)
        end
    end
    
    @testset "heapmin / heapmax tests" begin
        @testset "heapmin tests" begin
            @test heapmin([1]) == 1
            @test heapmin([1, 2]) == 1
            for i = 1:20
                A = rand(50)
                minmax_heapify!(A)
                @test heapmin(A) == minimum(A)
            end
        end
        @testset "heapmax tests" begin
            @test heapmax([1]) == 1
            @test heapmax([1, 2]) == 2
            @test heapmax([1, 3, 2]) == 3
            for i = 1:20
                A = rand(50)
                minmax_heapify!(A)
                @test heapmax(A) == maximum(A)
            end
        end
    end

end
