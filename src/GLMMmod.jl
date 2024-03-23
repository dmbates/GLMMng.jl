abstract type GLMMmod{T} <: StatsBase.RegressionModel end

"""
    pdeviance(m::GLMMmod)

Return the penalized deviance.

The deviance is the `StatsBase.deviance` value for the model without
random effects (i.e. as a GLM model) and the penalty is the squared
length of the vector of modes of the spherical random effects.
"""
function pdeviance(m::GLMMmod)
    return sum(m.tbl.dev) + sum(abs2, m.utbl.u)
end
