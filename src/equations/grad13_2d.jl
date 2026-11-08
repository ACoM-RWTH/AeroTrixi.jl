# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing-EC-Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    Grad13Equations2D(Kn; nu0 = 1)

Grad's 13-moment equations of rarefied gas dynamics in two space dimensions, in the
non-dimensional form of the finite-volume solver `MultiBlockMomentsFV` (see M. Torrilhon,
"Two-Dimensional Bulk Microflow Simulations Based on Regularized Grad's 13-Moment
Equations", Multiscale Model. Simul. 5 (2006), [DOI: 10.1137/050635444](https://doi.org/10.1137/050635444)).

The gas is described by the density ``\rho``, the velocity ``v``, the temperature
``\theta = p / \rho`` (in units of ``R T``), the full pressure tensor ``P_{ij} = \rho T_{ij}``
with ``T_{ij} = \theta \delta_{ij} + \sigma_{ij} / \rho``, and the heat flux ``q``. In two
dimensions, the pressure tensor is a three-dimensional object with ``P_{xz} = P_{yz} = 0`` and
``P_{zz} = 3 p - P_{xx} - P_{yy}``, so that the state has nine variables.

The nine conservative variables are the moments
```math
u = \left( \rho,\ \rho v_1,\ \rho v_2,\ E,\ m_{11},\ m_{12},\ m_{22},\ s_1,\ s_2 \right)
```
with the total energy ``E = \frac{1}{2} \rho |v|^2 + \frac{3}{2} \rho \theta``, the full second
moments ``m_{ij} = \rho (T_{ij} + v_i v_j)``, and the full energy fluxes
``s_i = q_i + E v_i + \rho (T_{i1} v_1 + T_{i2} v_2)``. The primitive variables are
``(\rho, v_1, v_2, \theta, T_{11}, T_{12}, T_{22}, q_1, q_2)``; see [`cons2moments`](@ref) for the
output variables ``(\rho, v, \theta, p, P_{ij}, q)`` of the finite-volume solver.

The fluxes use Grad's closure for the third moments, ``m_{ijk} = \frac{2}{5} (q_i \delta_{jk} +
q_j \delta_{ik} + q_k \delta_{ij}) / \rho``, and the contracted fourth moments
``R_{ij} = 7 \theta \sigma_{ij} / \rho + 5 \theta^2 \delta_{ij}``. The collision terms relax the
stress with the collision frequency ``\nu = \nu_0 \rho \sqrt{\theta} / \mathrm{Kn}`` (Maxwell
molecules) and the heat flux with ``\frac{2}{3} \nu``; they are provided as explicit source terms
by [`source_terms_grad13`](@ref). `Kn` is the Knudsen number and `nu0` the prefactor of the
collision frequency.

The equations are not strictly hyperbolic everywhere: far from equilibrium (large stress or heat
flux), real characteristic speeds are lost. The wave-speed estimates of this type use the
crude but safe bound ``2 \sqrt{\frac{5}{3} \theta}`` of the finite-volume solver for all
characteristic speeds; see [`max_abs_speeds`](@ref).

A specular (adiabatic, accommodation coefficient zero) wall is available as
[`boundary_condition_slip_wall`](@ref): the external state mirrors the interior state across the
wall, flipping the sign of all components that are odd in the wall-normal direction.
"""
struct Grad13Equations2D{RealT <: Real} <: AbstractEquations{2, 9}
    Kn  :: RealT   # Knudsen number
    nu0 :: RealT   # prefactor of the collision frequency ν = nu0 ρ √θ / Kn
end

function Grad13Equations2D(Kn; nu0 = 1)
    Kn_, nu0_ = promote(Kn, nu0)
    return Grad13Equations2D{typeof(Kn_)}(Kn_, nu0_)
end

function varnames(::typeof(cons2cons), ::Grad13Equations2D)
    return ("rho", "rho_v1", "rho_v2", "rho_e", "m11", "m12", "m22", "s1", "s2")
end

function varnames(::typeof(cons2prim), ::Grad13Equations2D)
    return ("rho", "v1", "v2", "theta", "T11", "T12", "T22", "q1", "q2")
end

# Default: no entropy is defined for Grad13, hence no entropy time derivative is computed
default_analysis_integrals(::Grad13Equations2D) = ()

###############################################################################
# Variable conversions

"""
    cons2prim(u, equations::Grad13Equations2D)

Convert the conservative variables `u` to the primitive variables
``(\\rho, v_1, v_2, \\theta, T_{11}, T_{12}, T_{22}, q_1, q_2)`` with the temperature
``\\theta = p / \\rho``, the specific pressure tensor ``T_{ij} = P_{ij} / \\rho``, and the heat
flux ``q``.
"""
@inline function cons2prim(u, equations::Grad13Equations2D)
    rho, rho_v1, rho_v2, E, m11, m12, m22, s1, s2 = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    vs = v1^2 + v2^2
    theta = (2 * E / rho - vs) / 3
    T11 = m11 / rho - v1^2
    T12 = m12 / rho - v1 * v2
    T22 = m22 / rho - v2^2
    q1 = s1 - E * v1 - rho * (T11 * v1 + T12 * v2)
    q2 = s2 - E * v2 - rho * (T12 * v1 + T22 * v2)

    return SVector(rho, v1, v2, theta, T11, T12, T22, q1, q2)
end

"""
    prim2cons(prim, equations::Grad13Equations2D)

Convert the primitive variables ``(\\rho, v_1, v_2, \\theta, T_{11}, T_{12}, T_{22}, q_1, q_2)``
to the conservative variables; see [`cons2prim`](@ref).
"""
@inline function prim2cons(prim, equations::Grad13Equations2D)
    rho, v1, v2, theta, T11, T12, T22, q1, q2 = prim

    vs = v1^2 + v2^2
    E = 0.5f0 * rho * vs + 1.5f0 * rho * theta

    return SVector(rho,
                   rho * v1,
                   rho * v2,
                   E,
                   rho * (T11 + v1^2),
                   rho * (T12 + v1 * v2),
                   rho * (T22 + v2^2),
                   q1 + E * v1 + rho * (T11 * v1 + T12 * v2),
                   q2 + E * v2 + rho * (T12 * v1 + T22 * v2))
end

"""
    cons2moments(u, equations::Grad13Equations2D)

Convert the conservative variables `u` to the ten output variables of the finite-volume
solver `MultiBlockMomentsFV`: ``(\\rho, v_1, v_2, \\theta, p, P_{11}, P_{12}, P_{22}, q_1, q_2)``
with the pressure ``p = \\rho \\theta`` and the pressure tensor ``P_{ij} = \\rho T_{ij}``.
Useful as `solution_variables` of the `SaveSolutionCallback` to compare with that solver.
"""
@inline function cons2moments(u, equations::Grad13Equations2D)
    rho, v1, v2, theta, T11, T12, T22, q1, q2 = cons2prim(u, equations)
    return SVector(rho, v1, v2, theta, rho * theta, rho * T11, rho * T12, rho * T22, q1,
                   q2)
end

function varnames(::typeof(cons2moments), ::Grad13Equations2D)
    return ("rho", "v1", "v2", "theta", "p", "P11", "P12", "P22", "q1", "q2")
end

@inline density(u, ::Grad13Equations2D) = u[1]

# velocity and temperature θ = p / ρ from the conservative variables
@inline function velocity_temperature(u, ::Grad13Equations2D)
    rho, rho_v1, rho_v2, E = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    theta = (2 * E / rho - (v1^2 + v2^2)) / 3
    return v1, v2, theta
end

"""
    temperature(u, equations::Grad13Equations2D)

The temperature ``\\theta = p / \\rho`` (in units of ``R T``).
"""
@inline function temperature(u, equations::Grad13Equations2D)
    _, _, theta = velocity_temperature(u, equations)
    return theta
end

@inline function pressure(u, equations::Grad13Equations2D)
    return u[1] * temperature(u, equations)
end

@inline function density_pressure(u, equations::Grad13Equations2D)
    return u[1] * pressure(u, equations)
end

@inline energy_total(u, ::Grad13Equations2D) = u[4]

@inline function energy_kinetic(u, equations::Grad13Equations2D)
    return 0.5f0 * (u[2]^2 + u[3]^2) / u[1]
end

@inline function energy_internal(u, equations::Grad13Equations2D)
    return energy_total(u, equations) - energy_kinetic(u, equations)
end

###############################################################################
# Stress, heat flux, and non-equilibrium measures

"""
    von_mises_stress(u, equations::Grad13Equations2D)

The von Mises stress ``\\sigma_\\mathrm{vM} = \\sqrt{\\frac{3}{2} \\sigma : \\sigma}`` of the
deviatoric stress ``\\sigma = P - p I`` with ``\\sigma_{zz} = -(\\sigma_{11} + \\sigma_{22})``.
"""
@inline function von_mises_stress(u, equations::Grad13Equations2D)
    rho, _, _, theta, T11, T12, T22, _, _ = cons2prim(u, equations)
    sigma11 = rho * (T11 - theta)
    sigma22 = rho * (T22 - theta)
    sigma12 = rho * T12
    sigmazz = -(sigma11 + sigma22)
    return sqrt(1.5f0 * (sigma11^2 + sigma22^2 + sigmazz^2 + 2 * sigma12^2))
end

"""
    nonequilibrium(u, equations::Grad13Equations2D)

Dimensionless measure of the deviation from local thermodynamic equilibrium,
```math
\\sqrt{ \\left(\\frac{\\sigma_\\mathrm{vM}}{p}\\right)^2 + \\frac{|q|^2}{p^2 \\theta} },
```
combining the von Mises stress (see [`von_mises_stress`](@ref)) relative to the pressure and the
heat flux relative to its scale ``p \\sqrt{\\theta}``. It is of the order of the Knudsen number in
smooth flow and of order one inside shock layers, and marks the regions where the moment
equations deviate from the Euler equations; it is meant as the `variable` of an `IndicatorMax`
for adaptive mesh refinement.
"""
@inline function nonequilibrium(u, equations::Grad13Equations2D)
    rho, _, _, theta, T11, T12, T22, q1, q2 = cons2prim(u, equations)
    p = rho * theta
    sigma11 = rho * (T11 - theta)
    sigma22 = rho * (T22 - theta)
    sigma12 = rho * T12
    sigmazz = -(sigma11 + sigma22)
    stress2 = 1.5f0 * (sigma11^2 + sigma22^2 + sigmazz^2 + 2 * sigma12^2)
    return sqrt((stress2 + (q1^2 + q2^2) / theta) / p^2)
end

"""
    collision_frequency(u, equations::Grad13Equations2D)

The collision frequency ``\\nu = \\nu_0 \\rho \\sqrt{\\theta} / \\mathrm{Kn}`` of the relaxation of
the stress; the heat flux relaxes with ``\\frac{2}{3} \\nu``.
"""
@inline function collision_frequency(u, equations::Grad13Equations2D)
    rho = u[1]
    theta = temperature(u, equations)
    return collision_frequency(rho, theta, equations)
end

@inline function collision_frequency(rho, theta, equations::Grad13Equations2D)
    return equations.nu0 * rho * sqrt(theta) / equations.Kn
end

###############################################################################
# Source terms (productions)

"""
    source_terms_grad13(u, x, t, equations::Grad13Equations2D)

The collision terms of Grad's 13-moment equations for Maxwell molecules as explicit source
terms: the stress relaxes as ``-\\nu \\sigma_{ij}`` and the heat flux as
``-\\frac{2}{3} \\nu q_i``, with the collision frequency ``\\nu`` of
[`collision_frequency`](@ref). Written in the conservative variables, the energy-flux equations
also receive the work of the stress relaxation, ``-\\nu \\sigma_{ij} v_j``.

The relaxation is stiff for small Knudsen numbers: an explicit time integration needs
``\\nu \\Delta t`` below the real-axis stability limit of the Runge-Kutta method, which can be
enforced with the [`RelaxationStepsizeCallback`](@ref).
"""
@inline function source_terms_grad13(u, x, t, equations::Grad13Equations2D)
    rho, v1, v2, theta, T11, T12, T22, q1, q2 = cons2prim(u, equations)
    nu = collision_frequency(rho, theta, equations)

    sigma11 = rho * (T11 - theta)
    sigma22 = rho * (T22 - theta)
    sigma12 = rho * T12
    z = zero(rho)

    return SVector(z, z, z, z,
                   -nu * sigma11,
                   -nu * sigma12,
                   -nu * sigma22,
                   -nu * (sigma11 * v1 + sigma12 * v2) - 2 * nu * q1 / 3,
                   -nu * (sigma12 * v1 + sigma22 * v2) - 2 * nu * q2 / 3)
end

###############################################################################
# Fluxes

# Grad13 flux in x-direction; the y-flux and fluxes in arbitrary directions are obtained by
# rotation (the equations are rotationally invariant).
@inline function flux_x(u, equations::Grad13Equations2D)
    rho, v1, v2, theta, T11, T12, T22, q1, q2 = cons2prim(u, equations)
    vs = v1^2 + v2^2
    E = u[4]

    # Grad13 closure: third moments m_ijk and contracted fourth moments R_ij (per density)
    M111 = 6 * q1 / (5 * rho)
    M121 = 2 * q2 / (5 * rho)  # = M112
    M221 = 2 * q1 / (5 * rho)  # = M122
    R11 = 5 * theta^2 + 7 * theta * (T11 - theta)
    R12 = 7 * theta * T12

    f1 = rho * v1
    f2 = rho * (T11 + v1^2)
    f3 = rho * (T12 + v1 * v2)
    f4 = E * v1 + rho * (T11 * v1 + T12 * v2) + q1
    f5 = rho * (M111 + 3 * T11 * v1 + v1^3)
    f6 = rho * (M121 + T11 * v2 + 2 * T12 * v1 + v1^2 * v2)
    f7 = rho * (M221 + T22 * v1 + 2 * T12 * v2 + v2^2 * v1)
    f8 = 2 * (q1 + rho * (T11 * v1 + T12 * v2)) * v1 + E * v1^2 +
         rho * (M111 * v1 + M121 * v2) + 0.5f0 * rho * (T11 * vs + R11)
    f9 = q2 * v1 + q1 * v2 + rho * (v1 * v2 * (T11 + T22) + 1.5f0 * T12 * vs) +
         E * v1 * v2 + rho * (M121 * v1 + M221 * v2) + 0.5f0 * rho * R12

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end

@inline function flux(u, orientation::Integer, equations::Grad13Equations2D)
    if orientation == 1
        return flux_x(u, equations)
    else # orientation == 2
        RealT = eltype(u)
        normal = SVector(zero(RealT), one(RealT))
        return rotate_from_x(flux_x(rotate_to_x(u, normal, equations), equations),
                             normal, equations)
    end
end

@inline function flux(u, normal_direction::AbstractVector, equations::Grad13Equations2D)
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_
    f = rotate_from_x(flux_x(rotate_to_x(u, normal, equations), equations), normal,
                      equations)
    return norm_ * f
end

"""
    rotate_to_x(u, normal_vector, equations::Grad13Equations2D)

Rotate the state `u` into the frame of the (normalized) `normal_vector` ``n`` and its tangent
``t = (-n_2, n_1)``: vectors ``w`` become ``(w \\cdot n, w \\cdot t)`` and the symmetric tensor
``m`` becomes ``(n \\cdot m n, t \\cdot m n, t \\cdot m t)``.
"""
@inline function rotate_to_x(u, normal_vector, equations::Grad13Equations2D)
    c = normal_vector[1]
    s = normal_vector[2]

    rho, rho_v1, rho_v2, E, m11, m12, m22, s1, s2 = u

    return SVector(rho,
                   c * rho_v1 + s * rho_v2,
                   -s * rho_v1 + c * rho_v2,
                   E,
                   c^2 * m11 + 2 * c * s * m12 + s^2 * m22,
                   -c * s * m11 + (c^2 - s^2) * m12 + c * s * m22,
                   s^2 * m11 - 2 * c * s * m12 + c^2 * m22,
                   c * s1 + s * s2,
                   -s * s1 + c * s2)
end

"""
    rotate_from_x(u, normal_vector, equations::Grad13Equations2D)

Inverse of [`rotate_to_x`](@ref).
"""
@inline function rotate_from_x(u, normal_vector, equations::Grad13Equations2D)
    c = normal_vector[1]
    s = normal_vector[2]

    rho, rho_vn, rho_vt, E, mnn, mnt, mtt, sn, st = u

    return SVector(rho,
                   c * rho_vn - s * rho_vt,
                   s * rho_vn + c * rho_vt,
                   E,
                   c^2 * mnn - 2 * c * s * mnt + s^2 * mtt,
                   c * s * mnn + (c^2 - s^2) * mnt - c * s * mtt,
                   s^2 * mnn + 2 * c * s * mnt + c^2 * mtt,
                   c * sn - s * st,
                   s * sn + c * st)
end

###############################################################################
# Wave speed estimates

# Crude but safe bound for all characteristic speeds relative to the flow, as used by the
# finite-volume solver: 2 √(5/3 θ) (the largest characteristic speed of Grad13 at equilibrium
# is about 2.13 √θ in 1D).
@inline function max_wave_speed_bound(theta)
    return 2 * sqrt(5 * theta / 3)
end

"""
    max_abs_speeds(u, equations::Grad13Equations2D)

Upper bounds ``|v_i| + 2 \\sqrt{\\frac{5}{3} \\theta}`` of the characteristic speeds in the
coordinate directions, used for the CFL condition.
"""
@inline function max_abs_speeds(u, equations::Grad13Equations2D)
    v1, v2, theta = velocity_temperature(u, equations)
    c = max_wave_speed_bound(theta)
    return abs(v1) + c, abs(v2) + c
end

@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::Grad13Equations2D)
    v1_ll, v2_ll, theta_ll = velocity_temperature(u_ll, equations)
    v1_rr, v2_rr, theta_rr = velocity_temperature(u_rr, equations)

    if orientation == 1
        v_ll, v_rr = v1_ll, v1_rr
    else # orientation == 2
        v_ll, v_rr = v2_ll, v2_rr
    end

    λ_min = v_ll - max_wave_speed_bound(theta_ll)
    λ_max = v_rr + max_wave_speed_bound(theta_rr)

    return λ_min, λ_max
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::Grad13Equations2D)
    v1_ll, v2_ll, theta_ll = velocity_temperature(u_ll, equations)
    v1_rr, v2_rr, theta_rr = velocity_temperature(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    norm_ = norm(normal_direction)

    λ_min = v_ll - max_wave_speed_bound(theta_ll) * norm_
    λ_max = v_rr + max_wave_speed_bound(theta_rr) * norm_

    return λ_min, λ_max
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::Grad13Equations2D)
    v1_ll, v2_ll, theta_ll = velocity_temperature(u_ll, equations)
    v1_rr, v2_rr, theta_rr = velocity_temperature(u_rr, equations)

    if orientation == 1
        v_ll, v_rr = v1_ll, v1_rr
    else # orientation == 2
        v_ll, v_rr = v2_ll, v2_rr
    end

    return max(abs(v_ll), abs(v_rr)) +
           max(max_wave_speed_bound(theta_ll), max_wave_speed_bound(theta_rr))
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::Grad13Equations2D)
    v1_ll, v2_ll, theta_ll = velocity_temperature(u_ll, equations)
    v1_rr, v2_rr, theta_rr = velocity_temperature(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    norm_ = norm(normal_direction)

    return max(abs(v_ll), abs(v_rr)) +
           max(max_wave_speed_bound(theta_ll), max_wave_speed_bound(theta_rr)) * norm_
end

@inline function max_abs_speed(u_ll, u_rr, orientation::Integer,
                               equations::Grad13Equations2D)
    v1_ll, v2_ll, theta_ll = velocity_temperature(u_ll, equations)
    v1_rr, v2_rr, theta_rr = velocity_temperature(u_rr, equations)

    if orientation == 1
        v_ll, v_rr = v1_ll, v1_rr
    else # orientation == 2
        v_ll, v_rr = v2_ll, v2_rr
    end

    return max(abs(v_ll) + max_wave_speed_bound(theta_ll),
               abs(v_rr) + max_wave_speed_bound(theta_rr))
end

@inline function max_abs_speed(u_ll, u_rr, normal_direction::AbstractVector,
                               equations::Grad13Equations2D)
    v1_ll, v2_ll, theta_ll = velocity_temperature(u_ll, equations)
    v1_rr, v2_rr, theta_rr = velocity_temperature(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
    norm_ = norm(normal_direction)

    return max(abs(v_ll) + max_wave_speed_bound(theta_ll) * norm_,
               abs(v_rr) + max_wave_speed_bound(theta_rr) * norm_)
end

###############################################################################
# HLL flux with clipped wave speeds

"""
    flux_hll_grad13(u_ll, u_rr, orientation_or_normal_direction, equations::Grad13Equations2D)

The HLL flux with the wave-speed bounds of [`min_max_speed_naive`](@ref), in the form of the
finite-volume solver `MultiBlockMomentsFV`: with the clipped speeds
``\\lambda^- = \\min(\\lambda_\\min, 0)`` and ``\\lambda^+ = \\max(\\lambda_\\max, 0)``,
```math
F = \\frac{\\lambda^+ f(u_L) - \\lambda^- f(u_R) + \\lambda^- \\lambda^+ (u_R - u_L)}{\\lambda^+ - \\lambda^-},
```
and the central flux ``\\frac{1}{2} (f(u_L) + f(u_R))`` at a stationary sonic state, where both
clipped speeds vanish. This differs from `FluxHLL(min_max_speed_naive)` only at that state,
where Trixi.jl returns ``f(u_L)``. The state occurs at a specular wall that is hit by a flow with
exactly the normal speed of the wave-speed bound, as the inflow state of the forward-facing
step: there, the central flux of the interior state and its mirror image is the wall flux,
whereas ``f(u_L)`` lets the gas flow out through the wall.
"""
@inline function flux_hll_grad13(u_ll, u_rr, orientation_or_normal_direction,
                                 equations::Grad13Equations2D)
    λ_min, λ_max = min_max_speed_naive(u_ll, u_rr, orientation_or_normal_direction,
                                       equations)
    λ_minus = min(λ_min, zero(λ_min))
    λ_plus = max(λ_max, zero(λ_max))

    f_ll = flux(u_ll, orientation_or_normal_direction, equations)
    f_rr = flux(u_rr, orientation_or_normal_direction, equations)

    if λ_plus - λ_minus > 0
        inv_λ_plus_minus_λ_minus = inv(λ_plus - λ_minus)
        factor_ll = λ_plus * inv_λ_plus_minus_λ_minus
        factor_rr = λ_minus * inv_λ_plus_minus_λ_minus
        factor_diff = λ_minus * λ_plus * inv_λ_plus_minus_λ_minus
        return factor_ll * f_ll - factor_rr * f_rr + factor_diff * (u_rr - u_ll)
    else
        return 0.5f0 * (f_ll + f_rr)
    end
end

###############################################################################
# Specular wall

"""
    specular_reflection(u, normal_vector, equations::Grad13Equations2D)

The state `u` mirrored across a wall with the (normalized) `normal_vector`: all components
that are odd in the wall-normal direction (normal momentum, the normal-tangential component of
the second moment, and the normal energy flux) change sign, even components are kept. This is
the external state of a specular wall with accommodation coefficient zero.
"""
@inline function specular_reflection(u, normal_vector, equations::Grad13Equations2D)
    u_local = rotate_to_x(u, normal_vector, equations)
    u_mirror_local = SVector(u_local[1], -u_local[2], u_local[3], u_local[4],
                             u_local[5], -u_local[6], u_local[7],
                             -u_local[8], u_local[9])
    return rotate_from_x(u_mirror_local, normal_vector, equations)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::Grad13Equations2D)

Specular wall (accommodation coefficient zero): the boundary flux is the numerical flux
between the interior state and its mirror image [`specular_reflection`](@ref). This matches the
reflected ghost-cell wall of the finite-volume solver `MultiBlockMomentsFV`. Used with
`P4estMesh` and `UnstructuredMesh2D`. The surface flux must handle the stationary sonic state
of a supersonic flow hitting the wall, see [`flux_hll_grad13`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              x, t,
                                              surface_flux_function,
                                              equations::Grad13Equations2D)
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    u_mirror = specular_reflection(u_inner, normal, equations)

    return surface_flux_function(u_inner, u_mirror, normal_direction, equations)
end

"""
    boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function, equations::Grad13Equations2D)

Specular wall for `TreeMesh`: the mirror image flips the sign of the components that are odd in
the coordinate direction `orientation` (see [`specular_reflection`](@ref)).
"""
@inline function boundary_condition_slip_wall(u_inner, orientation::Integer, direction,
                                              x, t,
                                              surface_flux_function,
                                              equations::Grad13Equations2D)
    RealT = eltype(u_inner)
    if orientation == 1
        normal = SVector(one(RealT), zero(RealT))
    else # orientation == 2
        normal = SVector(zero(RealT), one(RealT))
    end
    u_mirror = specular_reflection(u_inner, normal, equations)

    if iseven(direction) # u_inner is "left" of boundary, u_mirror is "right" of boundary
        flux = surface_flux_function(u_inner, u_mirror, orientation, equations)
    else # u_mirror is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_mirror, u_inner, orientation, equations)
    end

    return flux
end

###############################################################################
# Analysis: maximum of the collision frequency over all nodes

"""
    max_collision_frequency(u, mesh, equations::Grad13Equations2D, dg, cache)

Maximum of the [`collision_frequency`](@ref) over all nodes of the DG solution `u`. Can be
passed as `extra_analysis_integrals = (max_collision_frequency,)` to the `AnalysisCallback`;
together with the time step it shows the stiffness ``\\nu \\Delta t`` of the explicit source
terms.
"""
function max_collision_frequency(u, mesh, equations::Grad13Equations2D, dg, cache)
    nu_max = zero(real(dg))
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            nu_max = max(nu_max, collision_frequency(u_node, equations))
        end
    end
    return nu_max
end

function analyze(::typeof(max_collision_frequency), du, u, t, mesh,
                 equations::Grad13Equations2D, dg, cache)
    return max_collision_frequency(u, mesh, equations, dg, cache)
end

pretty_form_utf(::typeof(max_collision_frequency)) = "max ν"
pretty_form_ascii(::typeof(max_collision_frequency)) = "max_nu"
end # @muladd
