# tests for min-max heaps
using NearestNeighborDescent: is_minmax_heap, minmax_heapify!, heapmin, heapmax, heappop_min!, heappop_max!, mm_heappush!

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
    
    @testset "heappop tests" begin
        @testset "heappop_min tests" begin
            A = [1]
            @test heappop_min!(A) == 1
            @test length(A) == 0
            A = [1, 3, 2]
            @test heappop_min!(A) == 1
            @test length(A) == 2
            @test is_minmax_heap(A)
            @test heapmin(A) == minimum(A) == 2
            for i = 1:20
                A = rand(50)
                minmax_heapify!(A)
                minval = minimum(A)
                @test heappop_min!(A) == minval
                @test length(A) == 49
                @test is_minmax_heap(A)
                @test heapmin(A) == minimum(A)
            end
        end
        @testset "heappop_max tests" begin
            A = [1]
            @test heappop_max!(A) == 1
            @test length(A) == 0
            A = [1, 3, 2]
            @test heappop_max!(A) == 3
            @test length(A) == 2
            @test is_minmax_heap(A)
            @test heapmax(A) == maximum(A) == 2
            for i = 1:20
                A = rand(50)
                minmax_heapify!(A)
                maxval = maximum(A)
                @test heappop_max!(A) == maxval
                @test length(A) == 49
                @test is_minmax_heap(A)
                @test heapmax(A) == maximum(A)
            end
        end
    end
    
    @testset "mm_heappush! tests" begin
        A = []
        mm_heappush!(A, 1)
        @test A[1] == 1
        mm_heappush!(A, 2)
        @test is_minmax_heap(A)
        mm_heappush!(A, 10)
        @test is_minmax_heap(A)
        mm_heappush!(A, rand(Int)))
        @test is_minmax_heap(A)
        for i = 1:5
            A = rand(20)
            minmax_heapify!(A)
            for j = 1:4
                mm_heappush!(A, rand(Int))
                @test is_minmax_heap(A)
            end
        end
            
    end

end
