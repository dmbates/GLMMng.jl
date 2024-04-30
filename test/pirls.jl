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
    print(Table(m1.ytbl))
end

@testset "SSformula" begin
    d = datadict[:contra]
    f = @formula use ~ 1 + urban + ch * age + abs2(age)
    m1 = fit(GLMMmod, f, d, BernoulliLogit(), refarray(d.disturbn); contrasts)
    @test m1 isa SingleScalar
    @test deviance(m1) ≈ 2353.8241971081175
end

@testset "Goldstein" begin
    d = datadict[:goldstein]
    f = @formula y ~ 1
    m1 = fit(GLMMmod, f, d, PoissonLog(), refarray(d.group))
    @test m1 isa SingleScalar
end