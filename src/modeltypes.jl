struct SingleScalar{DL<:DistLink,T<:AbstractFloat,S<:Integer} <: GLMMmod{T}
    X::Matrix{T}
    θβ::Vector{T}
    ytbl::NamedTuple{                     # column-table
        (:refs, :y, :η, :offset),
        Tuple{Vector{S},Vector{T},Vector{T},Vector{T}},
    }
    utbl::NamedTuple{                     # column-table
        (:u, :u0, :Ldiag, :pdev, :pdev0, :aGHQ),
        NTuple{6,Vector{T}},
    }
    rtbl::Vector{                         # row-table
        NamedTuple{(:μ, :dev, :rtwwt, :wwres, :wwresp),NTuple{5,T}}
    }
end

function SingleScalar(
    DL::DistLink, X::AbstractMatrix{T}, y::AbstractVector{T}, refs::AbstractVector{S}
) where {T<:AbstractFloat,S<:Integer}
    X, y, refs = collect(X), collect(y), collect(refs)

    # use IRLS to check X and y, obtain initial β, and establish rtbl
    irls = fit!(GLM(DL, X, y); verbose=false)
    β = irls.β
    θβ = append!(ones(T, 1), β)       # initial θ = 1
    η = irls.ytbl.η

    # refs should contain all values from 1 to maximum(refs)
    refvals = sort!(unique(refs))
    q = length(refvals)
    if refvals ≠ 1:q
        throw(ArgumentError("sort!(unique(refs)) must be 1:$q"))
    end
    length(refs) == length(y) ||
        throw(ArgumentError("lengths of y and refs aren't equal"))

    ytbl = (; refs, y, η, offset=copy(η))

    utbl = NamedTuple(
        nm => zeros(T, q) for
        nm in (:u, :u0, :Ldiag, :pdev, :pdev0, :aGHQ)
    )
    return updatetbl!(SingleScalar{typeof(DL),T,S}(X, θβ, ytbl, utbl, irls.rtbl))
end

function fit(
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

function updatetbl!(m::SingleScalar{DL}) where {DL}
    (; refs, y, η, offset) = m.ytbl
    u = m.utbl.u
    θ = first(m.θβ)
    # evaluate η = offset + ZΛu where Λ is θ * I and Z is one-hot
    fill!(η, 0)
    @inbounds for i in eachindex(η, refs, offset)
        η[i] += muladd(θ, u[refs[i]], offset[i])
    end
    rtbl = m.rtbl
    @inbounds Threads.@threads for i in eachindex(rtbl, y, η, offset)
        rtbl[i] = tblrow(DL, y[i], η[i], offset[i])
    end
    return m
end

function updateu!(m::SingleScalar)
    (; u, u0, Ldiag) = m.utbl
    copyto!(u0, u)                # keep a copy of u
    θ = first(m.θβ)               # extract the scalar θ
    fill!(u, 0)
    if iszero(θ)                  # skip the update if θ == 0
        fill!(Ldiag, 1)           # L is the identity if θ == 0
        return updatetbl!(m)
    end
    fill!(Ldiag, 0)
    @inbounds for (ri, ti) in zip(m.ytbl.refs, m.rtbl)
        rtWΛ = θ * ti.rtwwt       # non-zero in i'th row of √WZΛ
        Ldiag[ri] += abs2(rtWΛ)   # accumulate Λ'Z'WZΛ
        u[ri] += rtWΛ * ti.wwresp # accumulate Λ'Z'Wỹ
    end
    Ldiag .+= 1                   # form diagonal of Λ'Z'WZΛ + I = LL'
    u ./= Ldiag                   # solve for u with diagonal LL'
    return updatetbl!(m)          # and update η and rtbl
end

function pirls!(m::SingleScalar; verbose::Bool=false)
    (; u, u0, Ldiag) = m.utbl
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
    map!(sqrt, Ldiag, Ldiag)      # replace diag(LL') by diag(L)
    return m
end

LinearAlgebra.logdet(m::SingleScalar) = 2 * sum(log, m.utbl.Ldiag)

function laplaceapprox(m::SingleScalar)
    return pdeviance(m) + logdet(m)
end

function pdevcomps!(m::SingleScalar)
    (; u, pdev) = m.utbl
    pdev .= abs2.(u)        # initialize pdevj to square of uj
    @inbounds for (ri, ti) in zip(m.ytbl.refs, m.rtbl)
        pdev[ri] += ti.dev
    end
    return m
end

function evalGHQ!(m::SingleScalar; nGHQ::Integer=11)
    (; ytbl, utbl, rtbl) = m
    (; u, u0, Ldiag, pdev, pdev0, aGHQ) = utbl
    pdevcomps!(pirls!(m))   # ensure that u0 and pdev0 are current
    copyto!(pdev0, pdev)
    fill!(aGHQ, 0)
    for (z, w) in GHnorm(nGHQ)
        if iszero(z)        # exp term is one when z == 0
            aGHQ .+= w
        else
            u .= u0 .+ z ./ Ldiag
            pdevcomps!(updatetbl!(m))
            aGHQ .+= w .* exp.((abs2(z) .+ pdev0 .- pdev) ./ 2)
        end
    end
    map!(log, aGHQ, aGHQ)   # log.(aGHQ) in place
    aGHQ .*= -2
    return m
end

function StatsAPI.fit!(m::SingleScalar; nGHQ::Integer=11)
    (; pdev0, aGHQ) = m.utbl
    θβ = m.θβ
    pp1 = length(θβ)        # length(β) = p and length(θ) = 1
    opt = Opt(:LN_BOBYQA, pp1)
    mβ = view(θβ, 2:pp1)
    function obj(x, g)
        if !isempty(g)
            throw(ArgumentError("gradient not provided, g must be empty"))
        end
        copyto!(θβ, x)
        mul!(m.ytbl.offset, m.X, mβ)
        evalGHQ!(m; nGHQ)
        return sum(pdev0) + sum(aGHQ) + logdet(m)
    end
    opt.min_objective = obj
    lb = fill!(similar(θβ), -Inf) # vector of lower bounds
    lb[1] = 0               # scalar θ must be non-negative
    NLopt.lower_bounds!(opt, lb)
    minf, minx, ret = optimize(opt, copy(θβ))
    @info (; ret, fevals=opt.numevals, minf)
    return m
end
