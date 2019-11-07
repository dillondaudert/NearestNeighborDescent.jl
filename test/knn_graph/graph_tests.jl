
@testset "HeapKNNGraph Tests" begin
    Random.seed!(0)
    small_data_f64 = [rand(5) for _ in 1:50]
    small_data_f32 = [rand(Float32, 5) for _ in 1:50]

    function is_valid_knn_graph(g::HeapKNNGraph{V, K}) where {V, K}
        for i in eachindex(g._knn_heaps)
            heap = g._knn_heaps[i]
            # check that all nodes have exactly K outgoing edges
            if length(heap) != K
                return false
            end
            # check that all edges start at the src for this heap
            for edge in heap.valtree
                if src(edge) != i
                    return false
                end
            end
            # check that the outgoing edge destinations are unique
            if length(heap) != length(Set(dst.(heap.valtree)))
                return false
            end
        end
        return true
    end

    @testset "Constructor Tests" begin
        # TODO: HeapKNNGraph(data, k, metric)
        @test_broken false
    end
    @testset "LightGraphs interface tests" begin
        # test LightGraphs interface methods exist on HeapKNNGraph
        for m in (edges, vertices, weights, edgetype, ne, nv, eltype)
            @test hasmethod(m, (HeapKNNGraph,))
        end
        @test hasmethod(has_edge, (HeapKNNGraph, Int, Int))
        @test hasmethod(has_edge, (HeapKNNGraph, HeapKNNGraphEdge))
        @test hasmethod(has_vertex, (HeapKNNGraph, Int))
        @test hasmethod(inneighbors, (HeapKNNGraph, Int))
        @test hasmethod(outneighbors, (HeapKNNGraph, Int))
        @test hasmethod(add_edge!, (HeapKNNGraph, HeapKNNGraphEdge))

        k = 10
        n = length(small_data_f64)
        g = HeapKNNGraph(small_data_f64, k, Euclidean())
        # TODO: 
        # edges
        @test length(collect(edges(g))) == n*k
        # vertices
        @test length(collect(vertices(g))) == n
        # weights
        @test size(weights(g)) == (n, n)
        @test sum(weights(g) .!= 0) == n*k
        # edgetype
        @test edgetype(g) <: HeapKNNGraphEdge
        # has_edge(g, s, d), has_edge(g, e)
        e, _ = iterate(edges(g))
        @test has_edge(g, src(e), dst(e))
        @test has_edge(g, e)
        # has_vertex
        @test has_vertex(g, 1)
        @test !has_vertex(g, -1)
        # ne
        @test ne(g) == n*k
        # nv
        @test nv(g) == n
        # outneighbors
        @test length(outneighbors(g, 1)) == k
        @test sizeof(outneighbors(g, 1)) == sizeof(eltype(g))*k
        # inneighbors
        @test eltype(inneighbors(g, 1)) == eltype(g)
        # add_edge!
        # case: fail to add bc it exists
        @test !(add_edge!(g, e))
        # case: fail to add edge bc it exists with a different weight
        e2 = HeapKNNGraphEdge(src(e), dst(e), rand(), flag(e))
        @test !(add_edge!(g, e2))
        w = weights(g)
        for _i in eachindex(w)
            global i = _i
            w[i] == 0 && i[1] != i[2] && break
        end
        e3 = HeapKNNGraphEdge(i[2], i[1], 0.0)
        @test !has_edge(g, e3)
        @test is_valid_knn_graph(g)
        @test add_edge!(g, e3)
        @test is_valid_knn_graph(g)
    end

end
