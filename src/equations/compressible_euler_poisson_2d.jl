# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing-EC-Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CompressibleEulerPoissonEquations2D(gamma; coupling = 0)

The compressible Euler equations in 2D, extended by an electrostatic potential ``\phi``
and the electric field ``E = -\nabla \phi``. The variables are
``(\rho, \rho v_1, \rho v_2, \rho e, \phi, E_1, E_2)``.

The Euler part is the one of `CompressibleEulerEquations2D(gamma)`. The potential and the
field have no flux and no numerical dissipation, so that they are not changed by the time
integration of the DG discretization. They are set by a callback that solves the Poisson
problem for ``\phi`` (see [`GridapCallback`](@ref) and [`write_electrostatic`](@ref)) and
act on the flow through the source terms [`source_terms_electrostatic`](@ref), scaled by
`coupling`.
"""
struct CompressibleEulerPoissonEquations2D{RealT <: Real,
                                           Euler <: CompressibleEulerEquations2D} <:
       AbstractEquations{2, 7}
    euler    :: Euler
    coupling :: RealT
end

function CompressibleEulerPoissonEquations2D(gamma; coupling = 0)
    euler = CompressibleEulerEquations2D(gamma)
    RealT = typeof(euler.gamma)
    return CompressibleEulerPoissonEquations2D{RealT, typeof(euler)}(euler,
                                                                     convert(RealT,
                                                                             coupling))
end

function varnames(::typeof(cons2cons), ::CompressibleEulerPoissonEquations2D)
    return ("rho", "rho_v1", "rho_v2", "rho_e_total", "phi", "E1", "E2")
end
function varnames(::typeof(cons2prim), ::CompressibleEulerPoissonEquations2D)
    return ("rho", "v1", "v2", "p", "phi", "E1", "E2")
end

# Euler part of the state and extension of Euler quantities by the electrostatic part
@inline euler_variables(u) = SVector(u[1], u[2], u[3], u[4])
@inline function with_electrostatic(euler_values, electrostatic_values)
    return SVector(euler_values[1], euler_values[2], euler_values[3], euler_values[4],
                   electrostatic_values[1], electrostatic_values[2],
                   electrostatic_values[3])
end
@inline function with_zero_electrostatic(euler_values)
    z = zero(eltype(euler_values))
    return with_electrostatic(euler_values, SVector(z, z, z))
end
@inline electrostatic_variables(u) = SVector(u[5], u[6], u[7])

@inline function flux(u, orientation_or_normal_direction,
                      equations::CompressibleEulerPoissonEquations2D)
    return with_zero_electrostatic(flux(euler_variables(u),
                                        orientation_or_normal_direction,
                                        equations.euler))
end

@inline function flux_ranocha(u_ll, u_rr, orientation_or_normal_direction,
                              equations::CompressibleEulerPoissonEquations2D)
    return with_zero_electrostatic(flux_ranocha(euler_variables(u_ll),
                                                euler_variables(u_rr),
                                                orientation_or_normal_direction,
                                                equations.euler))
end

# No numerical dissipation for the potential and the field
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr,
                                                              orientation_or_normal_direction,
                                                              equations::CompressibleEulerPoissonEquations2D)
    return with_zero_electrostatic(dissipation(euler_variables(u_ll),
                                               euler_variables(u_rr),
                                               orientation_or_normal_direction,
                                               equations.euler))
end

@inline function max_abs_speed(u_ll, u_rr, orientation_or_normal_direction,
                               equations::CompressibleEulerPoissonEquations2D)
    return max_abs_speed(euler_variables(u_ll), euler_variables(u_rr),
                         orientation_or_normal_direction, equations.euler)
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation_or_normal_direction,
                                     equations::CompressibleEulerPoissonEquations2D)
    return max_abs_speed_naive(euler_variables(u_ll), euler_variables(u_rr),
                               orientation_or_normal_direction, equations.euler)
end

@inline function max_abs_speeds(u, equations::CompressibleEulerPoissonEquations2D)
    return max_abs_speeds(euler_variables(u), equations.euler)
end

@inline function cons2prim(u, equations::CompressibleEulerPoissonEquations2D)
    return with_electrostatic(cons2prim(euler_variables(u), equations.euler),
                              electrostatic_variables(u))
end

@inline function prim2cons(prim, equations::CompressibleEulerPoissonEquations2D)
    return with_electrostatic(prim2cons(euler_variables(prim), equations.euler),
                              electrostatic_variables(prim))
end

# The potential and the field do not contribute to the entropy
@inline function cons2entropy(u, equations::CompressibleEulerPoissonEquations2D)
    return with_zero_electrostatic(cons2entropy(euler_variables(u), equations.euler))
end

@inline function entropy(u, equations::CompressibleEulerPoissonEquations2D)
    return entropy(euler_variables(u), equations.euler)
end

@inline density(u, ::CompressibleEulerPoissonEquations2D) = u[1]

@inline function pressure(u, equations::CompressibleEulerPoissonEquations2D)
    return pressure(euler_variables(u), equations.euler)
end

@inline function density_pressure(u, equations::CompressibleEulerPoissonEquations2D)
    return density_pressure(euler_variables(u), equations.euler)
end

@inline function energy_total(u, equations::CompressibleEulerPoissonEquations2D)
    return energy_total(euler_variables(u), equations.euler)
end

@doc raw"""
    source_terms_electrostatic(u, x, t, equations::CompressibleEulerPoissonEquations2D)

Force of the electric field on the flow, scaled by `equations.coupling`:
``c \rho E`` in the momentum equations and ``c \rho v \cdot E`` in the energy equation.
The potential and the field have no source terms; they are set by a callback.
"""
@inline function source_terms_electrostatic(u, x, t,
                                            equations::CompressibleEulerPoissonEquations2D)
    rho, rho_v1, rho_v2, _, _, E1, E2 = u
    c = equations.coupling
    z = zero(eltype(u))
    return SVector(z, c * rho * E1, c * rho * E2, c * (rho_v1 * E1 + rho_v2 * E2), z,
                   z, z)
end

@doc raw"""
    write_electrostatic(u, q, x, equations::CompressibleEulerPoissonEquations2D)

Write function of a [`GridapCallback`](@ref) solving for the potential ``\phi``: keep
the Euler variables of the state `u` and set the potential to ``\phi`` and the field to
``E = -\nabla \phi``, from the first solution field at the node `q`.
"""
@inline function write_electrostatic(u, q, x,
                                     equations::CompressibleEulerPoissonEquations2D)
    ∇φ = q2_gradient(q)
    return with_electrostatic(euler_variables(u), SVector(q2_value(q), -∇φ[1], -∇φ[2]))
end
end # @muladd
