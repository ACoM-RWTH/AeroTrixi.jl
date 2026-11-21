using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5 / 3
equations = CompressibleEulerEquations2D(gamma)

rho_ambient() = 0.5
p_ambient() = 0.4127

# Initial condition adopted from
# - Xiangxiong Zhang and Chi-Wang Shu (2011)
#   Positivity-preserving high order discontinuous Galerkin schemes
#   for compressible Euler equations with source terms
#   https://doi.org/10.1016/j.jcp.2010.10.036
# Mach 80 jet (with respect to the jet gas) entering the domain at x = 0
function initial_condition_astro_jet(x, t, equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    rho = rho_ambient()
    v1 = 0
    v2 = 0
    p = p_ambient()
    # add inflow for t > 0 at x = 0 and |y| < 0.05
    if (t > 0) && (x[1] ≈ 0) && (abs(x[2]) < RealT(0.05))
        rho = 5.0f0
        v1 = 30 # Mach number Ma ≈ 80 with respect to the jet gas
        v2 = 0
        p = p_ambient()
    end
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_astro_jet

boundary_conditions = (; x_neg = BoundaryConditionDirichlet(initial_condition_astro_jet),
                       x_pos = BoundaryConditionDirichlet(initial_condition_astro_jet),
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

# shock capturing necessary for this tough example
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.0001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

# Zhang and Shu (2011) use the domain [0, 2] x [-0.5, 0.5].
# Since the `TreeMesh` requires a square domain, we use [0, 2] x [-1, 1].
coordinates_min = (0.0, -1.0)
coordinates_max = (2.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                periodicity = (false, true))

# Simplified radiative cooling model used in section 3.4 of
# - Xiangxiong Zhang and Chi-Wang Shu (2011)
#   Positivity-preserving high order discontinuous Galerkin schemes
#   for compressible Euler equations with source terms
#   https://doi.org/10.1016/j.jcp.2010.10.036
# The energy equation obtains the sink
#   dE/dt = -Λ̃ (p² - p_a²) for T > T_a, and 0 otherwise,
# with ambient pressure p_a and ambient temperature T_a.
# The sink is clipped at zero, i.e., hot but rarefied gas (p < p_a) is not heated.
# Since T ∝ p / ρ, the condition T > T_a is evaluated as p / ρ > p_a / ρ_a.
# Λ̃ = 8.776 is the value given in
# - Y. Ha, C. Gardner, A. Gelb, C.-W. Shu (2005)
#   Numerical simulation of high Mach number astrophysical jets with radiative cooling
#   https://doi.org/10.1007/s10915-004-4786-4
# in their computational units.
struct SourceTermRadiativeCooling{RealT <: Real}
    Lambda_tilde::RealT
    p_ambient::RealT
    p_over_rho_ambient::RealT # ∝ ambient temperature
end

function SourceTermRadiativeCooling(; rho_ambient, p_ambient, Lambda_tilde = 8.776)
    Lambda_tilde, p_ambient, rho_ambient = promote(Lambda_tilde, p_ambient, rho_ambient)
    return SourceTermRadiativeCooling(Lambda_tilde, p_ambient, p_ambient / rho_ambient)
end

@inline function (source::SourceTermRadiativeCooling)(u, x, t,
                                                      equations::CompressibleEulerEquations2D)
    rho = u[1]
    p = pressure(u, equations)
    RealT = eltype(u)

    if p > source.p_over_rho_ambient * rho # T > T_a
        cooling = convert(RealT, source.Lambda_tilde * max(p^2 - source.p_ambient^2, 0))
    else
        cooling = zero(RealT)
    end

    return SVector(zero(RealT), zero(RealT), zero(RealT), -cooling)
end

# Ambient state
radiative_cooling = SourceTermRadiativeCooling(rho_ambient = rho_ambient(),
                                               p_ambient = p_ambient())

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = radiative_cooling)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.07)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# positivity limiter necessary for this tough example
positivity_limiter = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                         variables = (Trixi.density,
                                                                      pressure))

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 1.0,
                                          alpha_min = 0.0001,
                                          alpha_smooth = false,
                                          variable = Trixi.density)

amr_controller = ControllerThreeLevelCombined(semi, amr_indicator, indicator_sc,
                                              base_level = 2,
                                              med_level = 0, med_threshold = 0.0003, # med_level = current level
                                              max_level = 8, max_threshold = 0.003,
                                              max_threshold_secondary = indicator_sc.alpha_max)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true,
                           limiter! = positivity_limiter)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(; stage_limiter! = positivity_limiter, thread = Trixi.Threaded());
            ode_default_options()..., callback = callbacks);
