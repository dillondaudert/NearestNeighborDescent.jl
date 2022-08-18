
@testset "HeapKNNGraph Tests" begin
    Random.seed!(0)
    small_data_f64 = [rand(5) for _ in 1:50]
    small_data_f32 = [rand(Float32, 5) for _ in 1:50]

    function is_valid_knn_graph(g::ApproximateKNNGraph)
        for i in eachindex(g.heaps)
            heap = g.heaps[i]
            # check that all nodes have exactly K outgoing edges
            if length(heap) != g.n_neighbors
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
        for GraphT in [HeapKNNGraph, LockHeapKNNGraph]
            test_data = [rand(2) for _ in 1:4]
            test_metric = Euclidean()
            test_inds = Int[4 3 2 1; 2 1 4 3]
            test_dsts = Float64[1. 2. 3. 4.; 2. 3. 4. 5.]
            g = GraphT(test_data, test_metric, test_inds, test_dsts)
            @test g isa GraphT{Int, Float64, Vector{Vector{Float64}}, Euclidean}
            @test is_valid_knn_graph(g)

            # test construct from inds, dists matrices with matrix data input
            test_data_mat = rand(2, 4)
            DataViewT = typeof(collect(eachcol(test_data_mat)))
            g = GraphT(test_data_mat, test_metric, test_inds, test_dsts)
            @test g isa GraphT{Int, Float64, DataViewT, Euclidean}
            @test is_valid_knn_graph(g)

            @test_throws ErrorException GraphT(test_data, test_metric, test_inds, rand(3, 4))

            k = 10
            n = length(small_data_f64)
            g = GraphT(small_data_f64, k, Euclidean())
            @test g isa GraphT{Int, Float64, Vector{Vector{Float64}}}
            @test is_valid_knn_graph(g)

            # construct new KNNGraph with matrix data input
            g = GraphT(test_data_mat, 2, Euclidean())
            @test g isa GraphT{Int, Float64, DataViewT, Euclidean}
            @test is_valid_knn_graph(g)
        end
    end
    @testset "Graphs interface tests" begin
        # test Graphs interface methods exist on HeapKNNGraph, LockHeapKNNGraph
        for GraphT in [HeapKNNGraph, LockHeapKNNGraph]
            for m in (edges, vertices, weights, edgetype, ne, nv, eltype)
                @test hasmethod(m, (GraphT,))
            end
            @test hasmethod(has_edge, (GraphT, Int, Int))
            @test hasmethod(has_edge, (GraphT, HeapKNNGraphEdge))
            @test hasmethod(has_vertex, (GraphT, Int))
            @test hasmethod(inneighbors, (GraphT, Int))
            @test hasmethod(outneighbors, (GraphT, Int))
            @test hasmethod(add_edge!, (GraphT, HeapKNNGraphEdge))

            k = 10
            n = length(small_data_f64)
            g = GraphT(small_data_f64, k, Euclidean())
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
            # find two nodes without an edge
            function absent_edge(graph)
                # return the index to an edge that isn't in graph
                w = weights(graph)
                for i in eachindex(w)
                    w[i] == 0 && i[1] != i[2] && return i
                end
                error("Shouldn't get here")
            end
            ind = absent_edge(g)
            # test has_edge and add_edge!
            e3 = HeapKNNGraphEdge(ind[2], ind[1], 0.0)
            @test !has_edge(g, e3)
            @test is_valid_knn_graph(g)
            @test add_edge!(g, e3)
            @test is_valid_knn_graph(g)
        end
    end

    @testset "HeapKNNGraph utilities" begin
        for GraphT in [HeapKNNGraph, LockHeapKNNGraph]
            # knn_diameter
            @test hasmethod(knn_diameter, (GraphT{Int}, Int))
            # knn_matrices
            k = 10
            n = length(small_data_f64)
            g = GraphT(small_data_f64, k, Euclidean())
            @test hasmethod(knn_matrices, (GraphT,))
            inds, dists = knn_matrices(g)
            @test size(inds) == size(dists) == (k, n)
            @test all(issorted(col) for col in eachcol(dists))
            for i in 1:size(inds, 2), j_idx in 1:size(inds, 1)
                j = inds[j_idx, i]
                @test has_edge(g, i, j)
            end
            # edge_indices, node_edge
            @test length(edge_indices(g)) == n*k
            for ind in edge_indices(g)
                e = node_edge(g, ind[1], ind[2])
                @test e isa HeapKNNGraphEdge
                @test has_edge(g, e)
            end
            # node_edges
            @test length(node_edges(g, 1)) == k
            for e in node_edges(g, 1)
                @test src(e) == 1
                @test has_edge(g, e)
            end
            # update_flag!
            @test sum((!).(flag).(e for e in edges(g))) == 0 # all true flags
            @test flag(node_edge(g, 1, 1))
            new_e = update_flag!(g, 1, 1, false)
            @test !flag(new_e)
            @test new_e === node_edge(g, 1, 1)
            @test sum((!).(flag).(e for e in edges(g))) == 1 # one false flag
        end
    end

end
