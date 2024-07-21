@testset "SingleScalar" begin
    contra = datadict[:contra]
    f = @formula use ~ 1 + urban + ch * age + abs2(age)
    fsch = apply_schema(f, schema(f, contra, contrasts))
    resp, pred = modelcols(fsch, contra)
    m1 = SingleScalar(
        BernoulliLogit(),
        pred,
        vec(resp),
        refarray(contra.disturbn),
    )
    @test isone(first(m1.θβ))
    @test pdeviance(m1) ≈ 2303.284494736877
    @test pdeviance(GLMMng.updateu!(m1)) ≈ 2232.2174990169756
    @test pdeviance(GLMMng.pirls!(m1)) ≈ 2231.600219405606
    @test laplaceapprox(m1) ≈ 2373.518052747775
    @test length(coef(m1)) == 6
    @test dof(m1) == 7
    @test !isfitted(m1)
    fit!(m1)
    @test isfitted(m1)
    @test deviance(m1) ≈ 2353.8241971081175
end

@testset "Goldstein" begin
    d = datadict[:goldstein]
    f = @formula y ~ 1
    m1 = fit(GLMMmod, f, d, PoissonLog(), refarray(d.group))
    @test m1 isa SingleScalar
end

@testset "SingleScalarGoldstein" begin
    goldstein = datadict[:goldstein]
    f = @formula y ~ 1
    fsch = apply_schema(f, schema(f, goldstein, contrasts))
    resp, pred = modelcols(fsch, goldstein)
    m1 = SingleScalar(
        PoissonLog(),
        pred,
        float(vec(resp)),
        refarray(goldstein.group),
    )
    @test isone(first(m1.θβ))
    @test isa(fit!(m1; nGHQ=1), GLMMmod)
end

@testset "SSformula" begin
    d = datadict[:contra]
    f = @formula use ~ 1 + urban + ch * age + abs2(age)
    m1 = fit(GLMMmod, f, d, BernoulliLogit(), refarray(d.disturbn); contrasts)
    @test m1 isa SingleScalar
    @test deviance(m1) ≈ 2353.8241971081175
end

@testset "Lee" begin
    lee = dataset(:Lee)
    f = @formula(count ~ 1 + disease)
    fsch = apply_schema(f, schema(f, lee, contrasts))
    resp, pred = modelcols(fsch, lee)
    m2 = SingleScalar(
        PoissonLog(),
        pred,
        Vector{Float64}(resp),
        refarray(lee.var"Sample ID")
    )
    @test m2 isa SingleScalar{PoissonLog}
    fit!(m2; nGHQ=1)
    @test isfitted(m2)
    @test deviance(m2) ≈ 9873.398918968922
    @test coef(m2) ≈ [-0.842079325164301, 0.17094447151188819, -0.19644985451485605]
    @show length(m2.objectives)
end

@testset "Leeformula" begin
    d = datadict[:lee]
    f = @formula count ~ 1 + disease
    m1 = fit(GLMMmod, f, d, PoissonLog(), refarray(d.var"Sample ID"))
    @test m1 isa SingleScalar
#    @test dist(m1) isa Poisson
end
