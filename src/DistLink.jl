abstract type DistLink end
abstract type BernoulliLink <: DistLink end
abstract type PoissonLink <: DistLink end

# Make DistLinks broadcast like a scalar
Base.Broadcast.broadcastable(dl::DistLink) = Ref(dl)

struct BernoulliLogit <: BernoulliLink end  # FIXME: name conflicts with Distributions.BernoulliLogit
struct PoissonLog <: PoissonLink end

dist(::Union{DL,Type{DL}}) where {DL<:BernoulliLink} = Bernoulli()
dist(::Union{DL,Type{DL}}) where {DL<:PoissonLink} = Poisson()

_dispersion_parameter(::Bernoulli) = false
_dispersion_parameter(::Poisson) = false
_dispersion_parameter(dl::DistLink) = _dispersion_parameter(dist(dl))

function etastart(::Union{DL,Type{DL}}, y::T) where {DL<:BernoulliLink,T<:AbstractFloat}
    lthree = log(T(3))
    return ifelse(iszero(y), -lthree, lthree)
end

"""
    updatetbl!(tbl::MatrixTable, ::Union{BernoulliLogit,Type{BernoulliLogit})

Update the `μ`, `dev`, `rtwwt`, `wwresp` columns in `tbl`
"""
function updatetbl!(
    tbl::MatrixTable{Matrix{T}},
    ::Union{BernoulliLogit,Type{BernoulliLogit}},
) where {T<:AbstractFloat}
    (; y, offset, η, μ, dev, rtwwt, wwresp) = tbl
    @inbounds for i in axes(y, 1)
        ηi = η[i]
        yi = y[i]
        rtexpmη = exp(-ηi / 2)           # square root of exp(-ηi)
        expmη = abs2(rtexpmη)            # exp(-ηi)
        denom = one(T) + expmη
        μ[i] = μi = inv(denom)
        dev[i] = 2 * ((one(T) - yi) * ηi + log1p(expmη))
        rtwwt[i] = rtwwti = rtexpmη * μi # sqrt of working wt
        wwres = (yi - μi) / rtwwti       # weighted working resid
        wwresp[i] = wwres + rtwwti * (ηi - offset[i])
    end
    return tbl
end

function updatetbl!(tbl::MatrixTable{Matrix{T}},
    ::Union{PoissonLog,Type{PoissonLog}},
) where {T<:AbstractFloat}
    (; y, offset, η, μ, dev, rtwwt, wwresp) = tbl
    @inbounds for i in axes(y, 1)
        ηi = η[i]
        yi = y[i]
        μ[i] = μi = exp(η)
        dev[i] = 2 * (xlogy(yi, yi / μi) - (yi - μi))
        rtwwt[i] = rtwwti = one(T)   # placeholder - need to check the actual value
        wwresp = (yi - μi) / rtwwti + rtwwti * (ηi - offset[i])
    end
    return tbl
end