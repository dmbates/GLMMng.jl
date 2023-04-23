using Arrow
using GLMMng
using StatsModels
using Test

import DataAPI

const datadir = joinpath(@__DIR__, "..", "data")
dataset(nm::AbstractString) = Arrow.Table(joinpath(datadir, string(nm, ".arrow")))
dataset(nm::Symbol) = dataset(string(nm))

@testset "GLMBernoulli" begin
    admit = dataset(:admit)
    contrasts = Dict{Symbol,Any}(:rank => HelmertCoding())
    verbose = false
    f = @formula(admit ~ 1 + gre + gpa + rank)
    m1 = fit(GLM, f, admit, BernoulliLogit(); contrasts, verbose)
    @test m1 isa GLM{BernoulliLogit,Float64}
    @test length(m1.β) == 6
    @test m1.β ≈ [
        -4.8817570097706415,
        0.0022644256870777576,
        0.8040376543769175,
        -0.33772146507350154,
        -0.33416082173090234,
        -0.21989534752713139,
    ]
end

@testset "SingleScalar" begin
    contra = dataset(:contra)
    f = @formula use ~ 1 + urban + ch + age + abs2(age)
    contrasts = Dict{Symbol,Any}(:ch => HelmertCoding())
    fsch = apply_schema(f, schema(f, contra, contrasts))
    resp, pred = modelcols(fsch, contra)
    @info typeof(resp)
    m1 = SingleScalar(
        BernoulliLogit(),
        pred,
        vec(resp),
        DataAPI.refarray(contra.disturbn),
    )
    @test isone(first(m1.θβ))
    @test pdeviance(m1) ≈ 2417.864449812198
    @test pdeviance(GLMMng.updateu!(m1)) ≈ 2239.8473723836723
    @test pdeviance(GLMMng.pirls!(m1)) ≈ 2238.311233456473
    @test laplaceapprox(m1) ≈ 2380.586408514556
    fit!(m1)
    @info laplaceapprox(m1), sum(m1.utbl.aGHQ)
end

@testset "GHnorm" begin
    @test iszero(GHnorm(21)[11].abscissa)
    @test haskey(GLMMng.GHnormd, 21)
    @test 13 ≈ let μ = 2, σ = 3
        sum(v.weight * abs2(μ + σ * v.abscissa) for v in GHnorm(3))
    end
end