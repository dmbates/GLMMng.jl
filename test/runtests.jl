using Arrow
using GLMMng
using StatsModels
using Test

import DataAPI

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

@testset "GLMBernoulli" begin
    admit = datadict[:admit]
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
    contra = datadict[:contra]
    f = @formula use ~ 1 + urban + ch * age + abs2(age)
    fsch = apply_schema(f, schema(f, contra, contrasts))
    resp, pred = modelcols(fsch, contra)
    m1 = SingleScalar(
        BernoulliLogit(),
        pred,
        vec(resp),
        DataAPI.refarray(contra.disturbn),
    )
    @test isone(first(m1.θβ))
    @test pdeviance(m1) ≈ 2409.3774281600195
    @test pdeviance(GLMMng.updateu!(m1)) ≈ 2233.1209476972153
    @test pdeviance(GLMMng.pirls!(m1)) ≈ 2231.600219456821
    @test laplaceapprox(m1) ≈ 2373.5180529828467
    fit!(m1)
    @info laplaceapprox(m1), sum(m1.utbl.aGHQ)
end

@testset "SSformula" begin
    d = datadict[:contra]
    f = @formula use ~ 1 + urban + ch * age + abs2(age)
    m1 = fit(GLMMmod, f, d, BernoulliLogit(), DataAPI.refarray(d.disturbn); contrasts)
    @test m1 isa SingleScalar
end

@testset "GHnorm" begin
    @test iszero(GHnorm(21)[11].abscissa)
    @test haskey(GLMMng.GHnormd, 21)
    @test 13 ≈ let μ = 2, σ = 3
        sum(v.weight * abs2(μ + σ * v.abscissa) for v in GHnorm(3))
    end
end