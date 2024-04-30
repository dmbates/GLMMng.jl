
@testset "admitGlm" begin
    admit = datadict[:admit]
    f = @formula(admit ~ 1 + gre + gpa + rank)
    m1 = fit(Glm, f, admit, BernoulliLogit(); contrasts)
    @test m1 isa Glm{BernoulliLogit,Float64}
    @test length(m1.β) == 6
    @test dof(m1) == 6
    @test m1.β ≈ [
        -4.8817570097706415,
        0.0022644256870777576,
        0.8040376543769175,
        -0.33772146507350154,
        -0.33416082173090234,
        -0.21989534752713139,
    ]
    @test isfitted(m1)
    @test coef(m1) === m1.β
    @test deviance(m1) == last(m1.deviances)
    @test length(m1.deviances) ≤ 5
    @test nobs(m1) == length(m1.ytbl.y)
    @test loglikelihood(m1) == -deviance(m1) / 2
    @test length(response(m1)) == nobs(m1)
    @test response(m1) == admit.admit
    @test :QR ∈ propertynames(m1)
end

@testset "contraGlm" begin
    f = @formula use ~ 1 + urban + ch * age + abs2(age)
    m1 = fit(Glm, f, datadict[:contra], BernoulliLogit(); contrasts)
    @test m1 isa Glm{BernoulliLogit,Float64}
end

@testset "GoldsteinGlm" begin
    d = datadict[:goldstein]
    f = @formula y ~ 1   # mle should be mean(y)
    m1 = fit(Glm, f, d, PoissonLog())
    @test m1 isa Glm{PoissonLog,Float64}
    @test only(m1.β) ≈ log(mean(d.y))
end