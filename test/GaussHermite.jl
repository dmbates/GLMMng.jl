@testset "GHnorm" begin
    @test iszero(last(GHnorm(21).z))
    @test haskey(GLMMng.GHnormd, 21)
    @test 13 ≈ let μ = 2, σ = 3, GHn = GHnorm(3)
        sum(GHn.w .* abs2.(μ .+ σ .* GHn.z))
    end
end
