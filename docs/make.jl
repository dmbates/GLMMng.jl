using GLMMng
using Documenter

DocMeta.setdocmeta!(GLMMng, :DocTestSetup, :(using GLMMng); recursive=true)

makedocs(;
    modules=[GLMMng],
    authors="Douglas Bates <dmbates@gmail.com> and contributors",
    repo="https://github.com/dmbates/GLMMng.jl/blob/{commit}{path}#{line}",
    sitename="GLMMng.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dmbates.github.io/GLMMng.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dmbates/GLMMng.jl",
    devbranch="main",
)
