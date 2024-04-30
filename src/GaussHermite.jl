const header = (:z, :w)   # column names for MatrixTable
const GHnormd = Dict{Int, MatrixTable{Matrix{Float64}}}(
    1 => table([0. 1.]; header),
    2 => table([-1.0 0.5; 1.0 0.5]; header),
    3 => table([-sqrt(3) inv(6); sqrt(3) inv(6); 0. 2/3]; header)
)

"""
    GaussHermiteNormalized(k)

Returns a MatrixTable of `k` rows and columns `z`, the abscissae, and
`w`, the weights, of a `k`th order normalized Gauss-Hermite rule.

The rows are sorted by increasing value of `w` so the weighted sum adds the smallest magnitudes first.
"""
function GaussHermiteNormalized(k::Integer)
    ev = eigen(SymTridiagonal(zeros(k), sqrt.(1:(k - 1))))
    w = abs2.(ev.vectors[1, :])
    z = (ev.values .-= Iterators.reverse(ev.values)) ./ 2
    w .+= Iterators.reverse(w)
    normalize!(w ./= 2, 1)
    perm = sortperm(w)
    return table([z[perm] w[perm]]; header=(:z, :w))
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

