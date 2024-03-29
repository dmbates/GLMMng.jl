module GLMMng

import DataAPI: refarray, refpool
using Distributions: Bernoulli, Chisq, Poisson, ccdf, insupport
using LinearAlgebra: LinearAlgebra, Diagonal, SymTridiagonal, UpperTriangular
using LinearAlgebra: eigen, ldiv!, lmul!, logdet
using LinearAlgebra: mul!, norm, normalize!, qr!
using PRIMA: Status, bobyqa
using SparseArrays: SparseArrays
using StatsBase: StatsBase, CoefTable, aic, aicc, bic, coef, coeftable, deviance
using StatsBase: dof, dof_residual, fit, fit!, isfitted, loglikelihood
using StatsBase: meanresponse, nobs, response, stderror, vcov
using StatsModels: StatsModels, FormulaTerm, DummyCoding, HelmertCoding, EffectsCoding
using StatsModels: coefnames, @formula
using StatsModels: apply_schema, lrtest, modelcols, schema
using Tables: MatrixTable, table
using TypedTables: Table

include("DistLink.jl")
include("irls.jl")
include("GaussHermite.jl")
include("GLMMmod.jl")
include("modeltypes.jl")

export DistLink,
    Glm,
    GLMMmod,
    GHnorm,
    @formula,
    BernoulliLogit,
    DummyCoding,
    HelmertCoding,
    EffectsCoding,
    PoissonLog,
    SingleScalar,
    Table,
    deviance,
    coef,
    coeftable,
    dof,
    dof_residual,
    fit,
    fit!,
    isfitted,
    laplaceapprox,
    logdet,
    loglikelihood,
    lrtest,
    meanresponse,
    nobs,
    response,
    pdeviance,
    stderror,
    updatetbl!,
    refarray,
    refpool,
    vcov

end # module GLMMng
