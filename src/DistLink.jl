abstract type DistLink end
abstract type BernoulliLink <: DistLink end
abstract type PoissonLink <: DistLink end

# Make DistLinks broadcast like a scalar
Base.Broadcast.broadcastable(dl::DistLink) = Ref(dl)

struct BernoulliLogit <: BernoulliLink end  # FIXME: name conflicts with Distributions.BernoulliLogit
struct PoissonLog <: PoissonLink end

dist(::Union{DL,Type{DL}}) where {DL<:BernoulliLink} = Bernoulli()
dist(::Union{DL,Type{DL}}) where {DL<:PoissonLink} = Poisson()

function etastart(::Union{DL,Type{DL}}, y::T) where {DL<:BernoulliLink,T<:AbstractFloat}
    lthree = log(T(3))
    return ifelse(iszero(y), -lthree, lthree)
end

"""
    tblrow(::DistLink, y, η, offset=0)

Return a `NamedTuple` of `μ`, `dev`, `rtwwt`, `wwres`, `wwresp` from scalar `y`, `η`, and `offset`
"""
@inline function tblrow(
    ::Union{BernoulliLogit,Type{BernoulliLogit}},
    y::T,
    η::T,
    offset::T=zero(T)
) where {T<:AbstractFloat}
    rtexpmη = exp(-η / 2)      # square root of exp(-η)
    expmη = abs2(rtexpmη)      # exp(-η)
    denom = 1 + expmη
    μ = inv(denom)
    dev = 2 * ((1 - y) * η + log1p(expmη))
    rtwwt = rtexpmη / denom    # sqrt of working wt
    wwres = (y - μ) / rtwwt    # weighted working resid
    wwresp = wwres + rtwwt * (η - offset)
    return (; μ, dev, rtwwt, wwres, wwresp)
end

function tblrow(::PoissonLog, y::T, η::T, offset::T=zero(T)) where {T<:AbstractFloat}
    μ = exp(η)
    dev = 2 * (xlogy(y, y / μ) - (y - μ))
    rtwwt = one(T)   # placeholder - need to check the actual value
    wwres = (y - μ) / rtwwt
    wwresp = wwres + rtwwt * (η - offset)
    return (; μ, dev, rtwwt, wwres, wwresp)
end
