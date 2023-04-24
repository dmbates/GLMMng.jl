module GLMMng

using Distributions
using LinearAlgebra
using NLopt
using SparseArrays
using StatsAPI
using StatsModels
using Tables
using TypedTables

import StatsAPI: deviance, fit, fit!

include("GLMMmod.jl")
include("GaussHermite.jl")
include("DistLink.jl")
include("modeltypes.jl")
include("irls.jl")

export DistLink,
    GLM,
    GLMMmod,
    GHnorm,
    BernoulliLogit,
    PoissonLog,
    SingleScalar,
    deviance,
    fit,
    fit!,
    laplaceapprox,
    pdeviance

end # module GLMMng
