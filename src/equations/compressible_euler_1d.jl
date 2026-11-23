@muladd begin
#! format: noindent

# AUSM+ flux. Equation numbers refer to
# - Meng-Sing Liou (1996)
#   A Sequel to AUSM: AUSM+
#   [DOI: 10.1006/jcph.1996.0256](https://doi.org/10.1006/jcph.1996.0256)

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
end # @muladd
