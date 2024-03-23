struct SingleScalar{DL<:DistLink,T<:AbstractFloat,S<:Integer} <: GLMMmod{T}
    X::Matrix{T}
    θβ::Vector{T}
    refs::Vector{S}
    tbl::MatrixTable{Matrix{T}}
    utbl::MatrixTable{Matrix{T}}
    objectives::Vector{T}
end

function SingleScalar(
    DL::DistLink, X::AbstractMatrix{T}, y::AbstractVecOrMat{T}, refs::AbstractVector{S}
) where {T<:AbstractFloat,S<:Integer}
        # use Glm to check args and obtain a starting β
    irls = updateβ!(updateβ!(updateβ!(Glm(DL, X, y))))
    θβ = append!(ones(T, 1), irls.β)     # initial θ = 1
    tbl = irls.tbl
    copyto!(tbl.offset, tbl.η)           # offset is fixed-effects contribution

    refs = collect(refs)
    if length(refs) ≠ (n = length(y)) 
        throw(DimensionMismatch("length(y) = $n ≠ $(length(refs)) = length(refs)"))
    end
        # refs should contain all values from 1 to maximum(refs)
    refvals = sort!(unique(refs))
    q = length(refvals)
    if refvals ≠ 1:q
        throw(ArgumentError("sort!(unique(refs)) must be 1:$q"))
    end

    utbl = table(zeros(T, q, 6); header = (:u, :u0, :LLdiag, :pdev, :pdev0, :aGHQ))
    return updateu!(SingleScalar{typeof(DL),T,S}(X, θβ, refs, irls.tbl, utbl, T[]))
end

function evalGHQ!(m::SingleScalar; nGHQ::Integer=11)
    (; tbl, utbl) = m
    (; u, u0, LLdiag, pdev, pdev0, aGHQ) = utbl
    pdevcomps!(pirls!(m))   # ensure that u0 and pdev0 are current
    copyto!(pdev0, pdev)
    fill!(aGHQ, 0)
    for (z, w) in GHnorm(nGHQ)
        if iszero(z)        # exp term is one when z == 0
            aGHQ .+= w
        else
            u .= u0 .+ z ./ sqrt.(LLdiag)
            pdevcomps!(updatetbl!(m))
            aGHQ .+= w .* exp.((abs2(z) .+ pdev0 .- pdev) ./ 2)
        end
    end
    map!(log, aGHQ, aGHQ)   # log.(aGHQ) in place
    aGHQ .*= -2
    return m
end

function laplaceapprox(m::SingleScalar)
    return pdeviance(m) + logdet(m)
end

function pdevcomps!(m::SingleScalar)
    (; u, pdev) = m.utbl
    dev = m.tbl.dev
    pdev .= abs2.(u)        # initialize pdevj to square of uj
    @inbounds for (i, ri) in enumerate(m.refs)
        pdev[ri] += dev[i]
    end
    return m
end

function pirls!(m::SingleScalar; verbose::Bool=false)
    (; u, u0, LLdiag) = m.utbl
    fill!(u, 0)                   # start from u == 0
    copyto!(u0, u)                # keep a copy of u
    oldpdev = pdeviance(updatetbl!(m))
    verbose && @info 0, oldpdev
    for i in 1:10                 # maximum of 10 PIRLS iterations
        newpdev = pdeviance(updateu!(m))
        verbose && @info i, newpdev
        if newpdev > oldpdev      # PIRLS iteration failed
            @warn "PIRLS iteration did not reduce penalized deviance"
            copyto!(u, u0)        # restore previous u
            updatetbl!(m)         # restore η and rtbl
            break
        elseif (oldpdev - newpdev) < (1.0e-8 * oldpdev)
            copyto!(u0, u)        # keep a copy of u
            break
        else
            copyto!(u0, u)        # keep a copy of u
            oldpdev = newpdev
        end
    end
    return m
end

function updatetbl!(m::SingleScalar{DL}) where {DL}
    (; y, η, offset) = m.tbl
    refs = m.refs
    u = m.utbl.u
    θ = first(m.θβ)
    # evaluate η = offset + ZΛu where Λ is θ * I and Z is one-hot
    fill!(η, 0)
    @inbounds for i in axes(η, 1)
        η[i] += muladd(θ, u[refs[i]], offset[i])
    end
    updatetbl!(m.tbl, DL)
    return m
end

function updateu!(m::SingleScalar)
    (; refs, tbl, utbl) = m          # deconstruct m and utbl
    (; u, u0, LLdiag) = utbl
    (; rtwwt, wwresp) = tbl
    copyto!(u0, u)                   # keep a copy of u
    θ = first(m.θβ)                  # extract the scalar θ
    fill!(u, 0)                      # start u at zero
    if iszero(θ)                     # skip the update if θ == 0
        fill!(LLdiag, 1)             # L is the identity if θ == 0
    else
        fill!(LLdiag, 0)
        @inbounds for (i, r) in enumerate(refs)
            rtWΛ = θ * rtwwt[i]      # non-zero in i'th row of √WZΛ
            LLdiag[r] += abs2(rtWΛ)  # accumulate Λ'Z'WZΛ
            u[r] += rtWΛ * wwresp[i] # accumulate Λ'Z'Wỹ
        end
        LLdiag .+= 1                 # form diagonal of Λ'Z'WZΛ + I = LL'
        u ./= LLdiag                 # solve for u with diagonal LL'
    end
    return updatetbl!(m)             # and update η and tbl
end

LinearAlgebra.logdet(m::SingleScalar) = sum(log, m.utbl.LLdiag)

function StatsBase.coef(m::SingleScalar)
    θβ = m.θβ
    return view(θβ, 2:length(θβ))
end

function StatsBase.deviance(m::SingleScalar)
    obj = m.objectives
    isempty(obj) && throw(ArgumentError("model has not been fit"))
    return last(obj)
end

StatsBase.dof(m::SingleScalar) = length(m.θβ)

function StatsBase.fit(
    ::Type{GLMMmod},
    f::FormulaTerm,
    d,
    DL::DistLink,
    refs::Vector{S};
    contrasts::Dict{Symbol}=Dict{Symbol,Any}(),
    kwargs...,
) where {S<:Integer}
    fsch = apply_schema(f, schema(f, d, contrasts))
    resp, pred = modelcols(fsch, d)
    return fit!(SingleScalar(DL, pred, vec(resp), refs); kwargs...)
end

function StatsBase.fit!(m::SingleScalar; nGHQ::Integer=11)
    (; pdev0, aGHQ) = m.utbl
    θβ = m.θβ
    mβ = view(θβ, 2:length(θβ))
    lb = fill!(similar(θβ), -Inf) # vector of lower bounds
    lb[1] = 0                     # scalar θ must be non-negative
    objs = m.objectives
    function objective(x)
        copyto!(θβ, x)
        mul!(m.tbl.offset, m.X, mβ)
        evalGHQ!(m; nGHQ)
        obj = sum(pdev0) + sum(aGHQ) + logdet(m)
        push!(objs, obj)
        return obj
    end
    xmin, info = bobyqa(objective, copy(θβ); xl=lb, scale=abs.(θβ))
    issuccess(info) || throw(error("bobyqa returned status $(info.status) after $(info.nf) evaluations"))
    objective(xmin)
    return m
end

StatsBase.isfitted(m::SingleScalar) = !isempty(m.objectives)

#StatsBase.loglikelihood(m::) = -deviance(m) / 2

StatsBase.meanresponse(m::GLMMmod) = sum(response(m)) / nobs(m)

StatsBase.nobs(m::GLMMmod) = size(m.X, 1)

StatsBase.response(m::GLMMmod) = m.tbl.y
