struct Glm{DL<:DistLink,T<:AbstractFloat} <: StatsModels.RegressionModel
    form::Union{Nothing,FormulaTerm}
    X::AbstractMatrix{T}
    Xqr::Matrix{T}                # copy of X used for the QR decomposition
    ytbl::MatrixTable{Matrix{T}}
    Whalf::Diagonal{T}            # rtwwt as a Diagonal matrix
    β::Vector{T}
    βcp::Vector{T}
    deviances::Vector{T}
end

function Glm(
    DL::DistLink,
    X::AbstractMatrix{T},
    y::AbstractVecOrMat,
    form::Union{Nothing,FormulaTerm}=nothing,
) where {T<:AbstractFloat}
    D = dist(DL)
    all(insupport(D, y)) || throw(ArgumentError("Invalid y values for $(typeof(D))"))
    Xqr = copyto!(Matrix{T}(undef, size(X)), X)
    n = length(y)
    n ≠ size(X, 1) && throw(DimensionMismatch("size(X, 1) = $(size(X, 1)) ≠ $n = length(y)"))
    ytbl = table(zeros(T, n, 7); header=(:y, :offset, :η, :μ, :dev, :rtwwt, :wwresp))
    copyto!(ytbl.y, y)
    ytbl.η .= etastart.(DL, ytbl.y)
    updateytbl!(ytbl, DL)
    Whalf = Diagonal(ytbl.rtwwt)
    β = qr!(lmul!(Whalf, Xqr)) \ ytbl.wwresp 
    mul!(ytbl.η, X, β)
    updateytbl!(ytbl, DL)
    return Glm{typeof(DL), T}(form, X, Xqr, ytbl, Whalf, β, copy(β), T[])
end

"""
    updateβ!(m::Glm)

Utility function that saves the current `m.β` in `m.βcp` and evaluates a new `m.β` via weighted least squares.

After evaluating a new `m.β`, `m.ytbl` is updated
"""
function updateβ!(m::Glm{DL}) where {DL}
    (; X, Xqr, β, βcp, Whalf, ytbl) = m        # destructure m & ytbl
    (; η, wwresp) = ytbl
    copyto!(βcp, β)                            # keep a copy of β
    ldiv!(β, qr!(mul!(Xqr, Whalf, X)), wwresp) # weighted least squares
    mul!(η, X, β)                              # evaluate linear predictor
    updateytbl!(ytbl, DL)                      # update the rest of ytbl
    return m
end

function Base.getproperty(m::Glm, name::Symbol)
    if name == :QR
        return qr!(mul!(m.Xqr, m.Whalf, m.X))
    else
        return getfield(m, name)
    end
end

function Base.propertynames(m::Glm)
    return append!([:QR], fieldnames(typeof(m)))
end

function Base.show(io::IO, ::MIME"text/plain", m::Glm{DL}) where {DL}
    if isempty(m.deviances)
        @warn("Model has not been fit")
        return nothing
    end
    println(io, "Generalized Linear Model fit by maximum likelihood")
    println(io, "  ", m.form)
    println(io, "  DistributionLink: ", DL)
    println(io)
    nums = Base.Ryu.writefixed.([loglikelihood(m), deviance(m), aic(m), aicc(m), bic(m)], 4)
    fieldwd = max(maximum(textwidth.(nums)) + 1, 11)
    for label in [" logLik", " deviance", "AIC", "AICc", "BIC"]
        print(io, rpad(lpad(label, (fieldwd + textwidth(label)) >> 1), fieldwd))
    end
    println(io)
    print.(Ref(io), lpad.(nums, fieldwd))
    println(io)
    println(io)
    println(io, " Number of obs: ", nobs(m))

    println(io, "\nCoefficients:")
    return show(io, coeftable(m))
end

StatsBase.coef(m::Glm) = m.β

function StatsBase.coeftable(m::Glm)
    co = coef(m)
    se = stderror(m)
    z = co ./ se
    pvalue = ccdf.(Chisq(1), abs2.(z))
    p = length(z)
    names = isnothing(m.form) ? string.('x', lpad.(1:p, ndigits(p), '0')) : coefnames(m.form.rhs)

    return CoefTable(
        hcat(co, se, z, pvalue),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)"],
        names, 4, # pvalcol
        3, # teststatcol
    )
end

StatsBase.deviance(m::Glm) = sum(m.ytbl.dev)

StatsBase.dof(m::Glm) = size(m.X, 2)

function StatsBase.dof_residual(m::Glm)
    n, p = size(m.X)
    return n - p
end

function StatsBase.fit(
    ::Type{Glm},
    f::FormulaTerm,
    d,
    DL::DistLink;
    contrasts::Dict{Symbol}=Dict{Symbol,Any}(),
    kwargs...,
)
    fsch = apply_schema(f, schema(f, d, contrasts))
    resp, pred = modelcols(fsch, d)
    return fit!(Glm(DL, pred, resp, fsch); kwargs...)
end

function StatsBase.fit!(m::Glm{DL}, β₀=m.β) where {DL}
    (; X, β, βcp, ytbl, deviances) = m
    mul!(ytbl.η, X, copyto!(β, β₀))
    updateytbl!(ytbl, DL)
    olddev = deviance(m)
    push!(empty!(deviances), olddev)
    for i in 1:100                 # perform at most 100 iterations
        newdev = deviance(updateβ!(m))
        push!(deviances, newdev)
        if newdev > olddev
            @warn "failure to decrease deviance"
            copyto!(β, βcp)        # roll back changes to β, η, and ytbl
            mul!(ytbl.η, X, β)
            updateytbl!(ytbl, DL)
            break
        elseif (olddev - newdev) < (1.0e-10 * abs(olddev))
            break                  # exit loop if deviance is stable
        else
            olddev = newdev
            copyto!(βcp, β)
        end
    end
    return m
end

StatsBase.isfitted(m::Glm) = !isempty(m.deviances)

StatsBase.loglikelihood(m::Glm) = -deviance(m) / 2

StatsBase.meanresponse(m::Glm) = sum(response(m)) / nobs(m)

StatsBase.nobs(m::Glm) = size(m.X, 1)

StatsBase.response(m::Glm) = m.ytbl.y

function StatsBase.stderror(m::Glm)
    isempty(m.deviances) && throw(ArgumentError("model has not been fit"))
    return norm.(eachrow(inv(m.QR.R)))
end

function StatsBase.vcov(m::Glm)
    Rinv = inv(m.QR.R)
    return Rinv * Rinv'
end
