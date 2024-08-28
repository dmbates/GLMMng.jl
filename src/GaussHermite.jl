const GHnormd = let header = (:z, :w)   # cache results in a Dict
    Dict{Int,MatrixTable{Matrix{Float64}}}(
        1 => table([0.0 1.0]; header),
        2 => table([-1.0 0.5; 1.0 0.5]; header),
        3 => table([-sqrt(3) inv(6); sqrt(3) inv(6); 0.0 2/3]; header),
    )
end

"""
    GaussHermiteNormalized(k)

Returns a MatrixTable of `k` rows and columns `z`, the abscissae, and
`w`, the weights, of a `k`th order normalized Gauss-Hermite rule.

The rows are sorted by increasing value of `w` so the weighted sum adds the smallest magnitudes first.
"""
function GaussHermiteNormalized(k::Integer)
    val = table(Matrix{Float64}(undef, k, 2); header=(:z, :w))
    (; z, w) = val
    (; values, vectors) = eigen(SymTridiagonal(zeros(k), sqrt.(1:(k - 1))))
    z .= (values .- Iterators.reverse(values)) ./ 2
    values .= abs2.(vectors[1, :])
    w .= (values .+ Iterators.reverse(values))
    normalize!(w ./= 2, 1)
    perm = sortperm(w)
    permute!(w, perm)
    permute!(z, perm)
    return val
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
