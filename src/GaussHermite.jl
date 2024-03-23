const GHnormd = Dict{
    Int,
    Table{
        @NamedTuple{z::Float64, w::Float64},
        1,
        @NamedTuple{z::Vector{Float64}, w::Vector{Float64}}
        }
    }(
    1 => Table([(z=0.0, w=1.0)]),
    2 => Table([(z=-1.0, w=0.5), (z=1.0, w=0.5)]),
    3 => Table([
        (z=-sqrt(3), w=inv(6)),
        (z=0.0, w=2.0 / 3.0),
        (z=sqrt(3), w=inv(6)),
    ]),
)

function GaussHermiteNormalized(k::Integer)
    ev = eigen(SymTridiagonal(zeros(k), sqrt.(1:(k - 1))))
    w = abs2.(ev.vectors[1, :])
    return Table(
    (z=(ev.values .-= Iterators.reverse(ev.values)) ./ 2,
        w=normalize!((w .+= Iterators.reverse(w)) ./ 2, 1),
    )
)
end

"""
    GHnorm(k::Int)

Return a `k`-vector of ze and ws for normalized Gauss-Hermite quadrature

The function values are stored (memoized) when first evaluated.  Subsequent evaluations
for the same `k` have very low overhead.
"""
function GHnorm(k::Integer)
    return get!(GHnormd, k) do
        GaussHermiteNormalized(k)
    end
end

