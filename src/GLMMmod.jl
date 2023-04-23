abstract type GLMMmod{T} <: StatsAPI.RegressionModel end

"""
    pdeviance(m::GLMMmod)

Return the penalized deviance.

The deviance is the `StatsAPI.deviance` value for the model without
random effects (i.e. as a GLM model) and the penalty is the squared
length of the vector of modes of the spherical random effects.
"""
function pdeviance(m::GLMMmod)
    return sum(r.dev for r in m.rtbl) + sum(abs2, m.utbl.u)
end
