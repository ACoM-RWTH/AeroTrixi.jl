@muladd begin
#! format: noindent

# AUSM+ flux. Equation numbers refer to
# - Meng-Sing Liou (1996)
#   A Sequel to AUSM: AUSM+
#   [DOI: 10.1006/jcph.1996.0256](https://doi.org/10.1006/jcph.1996.0256)
# The low-Mach extension AUSM+-up is taken from
# - Meng-Sing Liou (2006)
#   A sequel to AUSM, Part II: AUSM+-up for all speeds
#   [DOI: 10.1016/j.jcp.2005.09.020](https://doi.org/10.1016/j.jcp.2005.09.020)

# Polynomial split Mach numbers M^±_(4) for |M| < 1, eq. (19b), with β = 1/8
@inline function split_mach_beta_plus(mach, beta = 0.125f0)
    # Note: In the original paper there is a typo (0.5 instead of 0.25)
    return 0.25f0 * (mach + 1)^2 + beta * (mach^2 - 1)^2
end
# eq. (19a)
@inline function split_mach_plus(mach, beta = 0.125f0)
    mach_abs = abs(mach)
    if mach_abs >= 1
        return 0.5f0 * (mach + mach_abs)
    else
        return split_mach_beta_plus(mach, beta)
    end
end

# eq. (19b)
@inline function split_mach_beta_minus(mach, beta = 0.125f0)
    # Note: In the original paper there is a typo (0.5 instead of 0.25)
    return -0.25f0 * (mach - 1)^2 - beta * (mach^2 - 1)^2
end
# eq. (19a)
@inline function split_mach_minus(mach, beta = 0.125f0)
    mach_abs = abs(mach)
    if mach_abs >= 1
        return 0.5f0 * (mach - mach_abs)
    else
        return split_mach_beta_minus(mach, beta)
    end
end

# Polynomial split pressure functions P^±_(5) for |M| < 1, with α = 3/16 in AUSM+
@inline function split_pressure_alpha_plus(mach, alpha = 0.1875f0)
    return 0.25f0 * (mach + 1)^2 * (2 - mach) + alpha * mach * (mach^2 - 1)^2
end
@inline function split_pressure_plus(mach, alpha = 0.1875f0)
    if abs(mach) >= 1
        return 0.5f0 * (1 + sign(mach))
    else
        return split_pressure_alpha_plus(mach, alpha)
    end
end

@inline function split_pressure_alpha_minus(mach, alpha = 0.1875f0)
    return 0.25f0 * (mach - 1)^2 * (2 + mach) - alpha * mach * (mach^2 - 1)^2
end
@inline function split_pressure_minus(mach, alpha = 0.1875f0)
    if abs(mach) >= 1
        return 0.5f0 * (1 - sign(mach))
    else
        return split_pressure_alpha_minus(mach, alpha)
    end
end

# Critical speed of sound a_star (a*), eq. (27)
@inline function a_star(rho, v1, p,
                        equations::CompressibleEulerEquations1D)
    gamma = equations.gamma

    a_squared = gamma * p / rho

    # This is a_star from eq. (27)
    return sqrt((2 * a_squared + (gamma - 1) * v1^2) / (gamma + 1))
end

# eq. (40)
@inline function a_tilde(a_star, u_mag)
    return a_star * a_star / max(a_star, u_mag)
end

# Interface speed of sound a_{1/2} of AUSM+, eq. (40), using |u_L| and |u_R|
@inline function a_interface_ausm(rho_ll, v1_ll, p_ll, rho_rr, v1_rr, p_rr,
                                  equations::CompressibleEulerEquations1D)
    a_tilde_ll = a_tilde(a_star(rho_ll, v1_ll, p_ll, equations), abs(v1_ll))
    a_tilde_rr = a_tilde(a_star(rho_rr, v1_rr, p_rr, equations), abs(v1_rr))
    return min(a_tilde_ll, a_tilde_rr)
end

# Assemble the AUSM+ flux from the interface Mach number `mach_interface` (m_{1/2})
# and pressure `p_interface` (p_{1/2}), eq. (A3).
# Splitting m_{1/2} into m^± is equivalent to upwinding the mass flux
# a_{1/2} m_{1/2} ρ_{L/R}.
"""
    flux_ausmplus(u_ll, u_rr, orientation::Integer,
                  equations::CompressibleEulerEquations1D)

Advection upstream splitting method AUSM+, which splits the flux into a convective part,
upwinded with an interface Mach number, and a pressure part.
This flux is not symmetric and can thus only be used as surface flux or in a finite volume
setting, not as volume flux in flux differencing.

References:
- Meng-Sing Liou (1996)
  A Sequel to AUSM: AUSM+
  [DOI: 10.1006/jcph.1996.0256](https://doi.org/10.1006/jcph.1996.0256)
"""
@inline function flux_ausmplus(u_ll, u_rr, orientation::Integer,
                               equations::CompressibleEulerEquations1D)
    rho_ll, v1_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, p_rr = cons2prim(u_rr, equations)

    a_interface = a_interface_ausm(rho_ll, v1_ll, p_ll, rho_rr, v1_rr, p_rr, equations)

    # (A1)
    mach_ll = v1_ll / a_interface # M_j
    mach_rr = v1_rr / a_interface # M_{j + 1}

    # (A2)
    mach_interface = split_mach_plus(mach_ll) + split_mach_minus(mach_rr) # m_{j + 1/2}
    p_interface = (split_pressure_plus(mach_ll) * p_ll +
                   split_pressure_minus(mach_rr) * p_rr) # p_{j + 1/2}

    mach_interface_abs = abs(mach_interface)
    mach_interface_plus = 0.5f0 * (mach_interface + mach_interface_abs) # m_{1/2}^+
    mach_interface_minus = 0.5f0 * (mach_interface - mach_interface_abs) # m_{1/2}^-

    # Convective quantities Φ = (ρ, ρ v, ρ H) with ρ H = ρ e + p
    phi_ll = SVector(u_ll[1], u_ll[2], u_ll[3] + p_ll)
    phi_rr = SVector(u_rr[1], u_rr[2], u_rr[3] + p_rr)

    # Convective part, upwinded with the interface Mach number
    flux_convective = a_interface *
                      (mach_interface_plus * phi_ll + mach_interface_minus * phi_rr)

    # Pressure part only contributes to the momentum equation
    flux_pressure = SVector(zero(p_interface), p_interface, zero(p_interface))

    return flux_convective + flux_pressure
end

# Interface speed of sound a_{1/2} of AUSM+-up, eq. (30) of Liou (2006), using the
# velocities signed with respect to the interface, i.e.,
# ã_L = a*_L^2 / max(a*_L, u_L) and ã_R = a*_R^2 / max(a*_R, -u_R).
# Thus, a* is only reduced for flow towards the interface. Compared to the absolute
# values in AUSM+, this avoids reducing a_{1/2} (and thus increasing the dissipation)
# in expansions. Both variants are consistent and invariant under reflection.
@inline function a_interface_ausm_signed(rho_ll, v1_ll, p_ll, rho_rr, v1_rr, p_rr,
                                         equations::CompressibleEulerEquations1D)
    a_tilde_ll = a_tilde(a_star(rho_ll, v1_ll, p_ll, equations), v1_ll)
    a_tilde_rr = a_tilde(a_star(rho_rr, v1_rr, p_rr, equations), -v1_rr)
    return min(a_tilde_ll, a_tilde_rr)
end

"""
    FluxAUSMPlusUp(; mach_inf = 1, K_p = 0.25, K_u = 0.75, sigma = 1)

AUSM+-up flux of Liou (2006), an extension of [`flux_ausmplus`](@ref) to all speeds.
Compared to AUSM+, it adds
- a pressure diffusion term ``M_p \\propto -(p_R - p_L)`` to the interface Mach number,
  scaled by `K_p` and `sigma`, and
- a velocity diffusion term ``p_u \\propto -(u_R - u_L)`` to the interface pressure,
  scaled by `K_u`.
The reference Mach number `mach_inf` (``M_\\infty > 0``) sets the cutoff of the low-Mach
scaling function ``f_a(M_o) = M_o (2 - M_o)`` with
``M_o^2 = \\min(1, \\max(\\bar{M}^2, M_\\infty^2))``.
The default `mach_inf = 1` gives ``f_a = 1``, i.e., no low-Mach (preconditioning) scaling.

The default instance `flux_ausmplusup = FluxAUSMPlusUp()` is available as well.
This flux is not symmetric and can thus only be used as surface flux or in a finite volume
setting, not as volume flux in flux differencing.

References:
- Meng-Sing Liou (2006)
  A sequel to AUSM, Part II: AUSM+-up for all speeds
  [DOI: 10.1016/j.jcp.2005.09.020](https://doi.org/10.1016/j.jcp.2005.09.020)
"""
struct FluxAUSMPlusUp{RealT <: Real}
    mach_inf::RealT # reference/cutoff Mach number M_∞
    K_p::RealT # pressure diffusion coefficient
    K_u::RealT # velocity diffusion coefficient
    sigma::RealT # threshold of the pressure diffusion term
end

# The defaults are exactly representable `Float32` constants such that the flux does not
# promote `Float32` states to `Float64`
function FluxAUSMPlusUp(; mach_inf = 1.0f0, K_p = 0.25f0, K_u = 0.75f0, sigma = 1.0f0)
    if !(mach_inf > 0)
        throw(ArgumentError("The reference Mach number `mach_inf` must be positive."))
    end
    return FluxAUSMPlusUp(promote(mach_inf, K_p, K_u, sigma)...)
end

const flux_ausmplusup = FluxAUSMPlusUp()

@inline function (flux_ausm::FluxAUSMPlusUp)(u_ll, u_rr, orientation::Integer,
                                             equations::CompressibleEulerEquations1D)
    (; mach_inf, K_p, K_u, sigma) = flux_ausm

    rho_ll, v1_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, p_rr = cons2prim(u_rr, equations)

    a_interface = a_interface_ausm_signed(rho_ll, v1_ll, p_ll, rho_rr, v1_rr, p_rr,
                                          equations)

    mach_ll = v1_ll / a_interface # M_L
    mach_rr = v1_rr / a_interface # M_R

    # Low-Mach scaling function f_a(M_o) = M_o (2 - M_o)
    mach_mean_sqr = (v1_ll^2 + v1_rr^2) / (2 * a_interface^2) # \bar{M}^2
    mach_o = sqrt(min(1, max(mach_mean_sqr, mach_inf^2))) # M_o
    f_a = mach_o * (2 - mach_o)
    # Parameter of the split pressure functions in AUSM+-up, reduces to 3/16 for f_a = 1
    alpha = 0.1875f0 * (-4 + 5 * f_a^2)

    rho_interface = 0.5f0 * (rho_ll + rho_rr) # ρ_{1/2}

    # Interface Mach number with pressure diffusion term M_p
    mach_p = (-K_p / f_a * max(1 - sigma * mach_mean_sqr, 0) * (p_rr - p_ll) /
              (rho_interface * a_interface^2))
    mach_interface = split_mach_plus(mach_ll) + split_mach_minus(mach_rr) + mach_p

    # Interface pressure with velocity diffusion term p_u. Note that ρ_L + ρ_R is not
    # divided by 2. Since P^± >= 0, p_u increases the pressure under compression.
    p_plus = split_pressure_plus(mach_ll, alpha) # \mathcal{P}^+(M_L)
    p_minus = split_pressure_minus(mach_rr, alpha) # \mathcal{P}^-(M_R)
    p_u = (-K_u * p_plus * p_minus * (rho_ll + rho_rr) * f_a * a_interface *
           (v1_rr - v1_ll))
    p_interface = p_plus * p_ll + p_minus * p_rr + p_u

    mach_interface_abs = abs(mach_interface)
    mach_interface_plus = 0.5f0 * (mach_interface + mach_interface_abs) # m_{1/2}^+
    mach_interface_minus = 0.5f0 * (mach_interface - mach_interface_abs) # m_{1/2}^-

    # Convective quantities Φ = (ρ, ρ v, ρ H) with ρ H = ρ e + p
    phi_ll = SVector(u_ll[1], u_ll[2], u_ll[3] + p_ll)
    phi_rr = SVector(u_rr[1], u_rr[2], u_rr[3] + p_rr)

    # Convective part, upwinded with the interface Mach number
    flux_convective = a_interface *
                      (mach_interface_plus * phi_ll + mach_interface_minus * phi_rr)

    # Pressure part only contributes to the momentum equation
    flux_pressure = SVector(zero(p_interface), p_interface, zero(p_interface))

    return flux_convective + flux_pressure
end
end # @muladd
