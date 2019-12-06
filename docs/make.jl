using Documenter, NearestNeighborDescent

makedocs(
    sitename="NearestNeighborDescent.jl",
    modules=[NearestNeighborDescent],
    pages = [
        "Home" => "index.md",
        "Basic Usage" => "basic.md",
        "KNN Graphs" => [
            "graphs.md",
        ]
    ],
)

deploydocs(
    repo="github.com/dillondaudert/NearestNeighborDescent.jl.git",
    devbranch="v0.3.0-dev",
)
