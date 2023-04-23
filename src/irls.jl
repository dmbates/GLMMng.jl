struct GLM{DL<:DistLink,T<:AbstractFloat}
    X::Matrix{T}
    Xqr::Matrix{T}                # copy of X used in the QR decomp
    β::Vector{T}
    βcp::Vector{T}                # copy of previous β
    Whalf::Diagonal{T,Vector{T}}  # rtwwt as a Diagonal matrix
    ytbl::NamedTuple{(:y, :η),NTuple{2,Vector{T}}}
    rtbl::Vector{
        NamedTuple{(:μ, :dev, :rtwwt, :wwres, :wwresp),NTuple{5,T}}
    }
end

function GLM(
    DL::DistLink,
    X::AbstractMatrix{T},
    y::AbstractVector{S}
) where {T<:AbstractFloat,S}
    D = dist(DL)
    all(insupport(D, y)) || throw(ArgumentError("Invalid y values for $(typeof(D))"))
    Tprime = promote_type(T, S)
    X = collect(Tprime, X)
    y = collect(Tprime, y)

    n, p = size(X)
    if length(y) ≠ n
        throw(DimensionMismatch("length(y) = $(length(y)) ≠ $n = size(X, 1)"))
    end
    η = etastart.(DL, y)
    Xqr = copy(X)
    β = fill(-zero(T), p)
    βcp = copy(β)
    rtbl = tblrow.(DL, y, η)
    Whalf = Diagonal([r.rtwwt for r in rtbl])
    return GLM{typeof(DL),T}(X, Xqr, β, βcp, Whalf, (; y, η), rtbl)
end

deviance(glm::GLM) = sum(r.dev for r in glm.rtbl)

function updateβ!(m::GLM{DL}) where {DL}
    (; X, Xqr, β, βcp, Whalf, ytbl, rtbl) = m  # destructure m & ytbl
    (; y, η) = ytbl
    copyto!(βcp, β)                            # keep a copy of β
    copyto!(Whalf.diag, r.rtwwt for r in rtbl) # rtwwt -> Whalf
    mul!(Xqr, Whalf, X)                        # weighted model matrix
    copyto!(η, r.wwresp for r in rtbl)         # use η as temp storage
    ldiv!(β, qr!(Xqr), η)                      # weighted least squares
    rtbl .= tblrow.(DL, y, mul!(η, X, β))      # update η and rtbl
    return m
end

function fit!(m::GLM{DL}, β₀=m.β; verbose::Bool=true) where {DL}
    (; X, β, βcp, ytbl, rtbl) = m
    (; y, η) = ytbl
    rtbl .= tblrow.(DL, y, mul!(η, X, copyto!(β, β₀)))
    olddev = deviance(m)
    verbose && @info 0, olddev     # record the deviance at initial β
    for i in 1:100                 # perform at most 100 iterations
        newdev = deviance(updateβ!(m))
        verbose && @info i, newdev # iteration number and deviance
        if newdev > olddev
            @warn "failure to decrease deviance"
            copyto!(β, βcp)        # roll back changes to β, η, and rtbl
            rtbl = tblrow.(DL, y, mul!(η, X, β))
            break
        elseif (olddev - newdev) < (1.0e-10 * olddev)
            break                  # exit loop if deviance is stable
        else
            olddev = newdev
        end
    end
    return m
end

function fit(
    ::Type{GLM},
    f::FormulaTerm,
    d,
    DL::DistLink;
    contrasts::Dict{Symbol}=Dict{Symbol,Any}(),
    kwargs...,
)
    fsch = apply_schema(f, schema(f, d, contrasts))
    resp, pred = modelcols(fsch, d)
    return fit!(GLM(DL, pred, resp); kwargs...)
end
