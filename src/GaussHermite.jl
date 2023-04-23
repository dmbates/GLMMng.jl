const GHnormd = Dict{Int,Vector{NamedTuple{(:abscissa, :weight),NTuple{2,Float64}}}}(
    1 => [(abscissa=0.0, weight=1.0)],
    2 => [(abscissa=-1.0, weight=0.5), (abscissa=1.0, weight=0.5)],
    3 => [
        (abscissa=-sqrt(3), weight=inv(6)),
        (abscissa=0.0, weight=2.0 / 3.0),
        (abscissa=sqrt(3), weight=inv(6)),
    ],
)

function GaussHermiteNormalized(k::Integer)
    ev = eigen(SymTridiagonal(zeros(k), sqrt.(1:(k - 1))))
    w = abs2.(ev.vectors[1, :])
    return rowtable(
    (abscissa=(ev.values .-= reverse(ev.values)) ./ 2,
        weight=normalize!((w .+= reverse(w)) ./ 2, 1),  # weights should sum to 1
    )
)
end

"""
    GHnorm(k::Int)

Return a `k`-vector of abscissae and weights for normalized Gauss-Hermite quadrature

The function values are stored (memoized) when first evaluated.  Subsequent evaluations
for the same `k` have very low overhead.
"""
function GHnorm(k::Int)
    return get!(GHnormd, k) do
        GaussHermiteNormalized(k)
    end
end

GHnorm(k) = GHnorm(Int(k))
