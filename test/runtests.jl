using Arrow, GLMMng, LinearAlgebra, StatsModels, Test, TypedTables

const datadir = joinpath(@__DIR__, "..", "data")
dataset(nm::Symbol) = Arrow.Table(joinpath(datadir, string(nm, ".arrow")))
const datadict = Dict{Symbol,Arrow.Table}(
    :admit => dataset(:admit),
    :contra => dataset(:contra),
)

const contrasts = Dict{Symbol,Any}(
    :ch => HelmertCoding(),
    :rank => HelmertCoding(),
    :urban => HelmertCoding(),
)

include("./irls.jl")
include("./GaussHermite.jl")
include("./pirls.jl")