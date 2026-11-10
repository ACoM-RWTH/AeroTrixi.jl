using OrdinaryDiffEqLowStorageRK
using Trixi
using AeroTrixi

###############################################################################
# Supersonic flow of a rarefied gas over a forward-facing step, described by Grad's
# 13-moment equations (Grad13). The setup reproduces the case `cases/ffs.txt` of the
# finite-volume solver `MultiBlockMomentsFV` (M. Torrilhon): the channel [0, 3] × [0, 1] with
# a step of height 1/3 at x = 1, an inflow with density 1, temperature 1 and velocity
# 2 √(5/3) (Mach 2 with respect to the Euler sound speed √(5/3 θ)), and specular walls.
#
# The DG discretization uses the Lax-Friedrichs flux with the wave-speed bounds of the
# finite-volume solver, shock capturing by blending with a subcell finite-volume scheme, and adaptive mesh
# refinement driven by the local deviation from equilibrium, `nonequilibrium`: the stress and
# the heat flux relative to the pressure. This measure is of the order of the Knudsen number in
# smooth flow and of order one in the shock layers, so the refinement follows the shocks and
# the shear layers of the moment solution and converges once their inner structure is resolved.
#
# The collision terms are explicit source terms. They relax the stress with the collision
# frequency ν = ρ √θ / Kn (about 6000 behind the shock at Kn = 0.001), so the time step has to
# keep ν Δt below the real-axis stability limit of the Runge-Kutta method (about 2.5) in
# addition to the CFL condition. The time step is controlled by the CFL number alone; the
# analysis output prints the maximum of ν to judge ν Δt. With the refinement below, the CFL
# time step keeps ν Δt at about 1; coarser meshes need a smaller CFL number.

Kn = 0.005
equations = Grad13Equations2D(Kn)

"""
    initial_condition_forward_facing_step(x, t, equations::Grad13Equations2D)

Uniform supersonic flow in equilibrium (no stress, no heat flux) with density 1, temperature 1,
and velocity 2 √(5/3) in x-direction. Also used as the inflow state.
"""
function initial_condition_forward_facing_step(x, t, equations::Grad13Equations2D)
    rho = 1.0
    v1 = 2 * sqrt(5 / 3)
    v2 = 0.0
    theta = 1.0
    # equilibrium: T_ij = θ δ_ij, q = 0
    return prim2cons(SVector(rho, v1, v2, theta, theta, 0.0, theta, 0.0, 0.0), equations)
end
initial_condition = initial_condition_forward_facing_step

# Lax-Friedrichs flux with the crude wave-speed bounds of the finite-volume solver. The shock
# in front of the step is a reflected shock with a relative Mach number of 3, far beyond the
# critical Mach number of Grad13, so that the moment equations are not hyperbolic inside the
# shock layer. With the less dissipative HLL flux (`flux_hll_grad13`), a density spike grows at
# the head of the shock and the run breaks down; the additional dissipation of the
# Lax-Friedrichs flux keeps the shock layer at the values of the finite-volume solution.
#
# Shock capturing: there is no entropy-conservative two-point flux for Grad13, so the DG volume
# integral uses the central flux (standard DGSEM), and troubled elements (the non-equilibrium
# indicator below) are blended with a second-order subcell finite-volume scheme on the LGL subcells
# with slopes in the primitive variables limited by the van Leer limiter, the counterpart of the
# MUSCL scheme of the finite-volume solver (Rueda-Ramírez et al.). The first-order blending
# `VolumeIntegralShockCapturingHG` is more robust but smears the reflected shock and the shear
# layer; the pure DG method (`VolumeIntegralWeakForm`) breaks down in the shock layers. The
# limited subcell scheme alone, without the DG part, is the finite-volume solver on the subcells:
#   volume_integral = VolumeIntegralPureLGLFiniteVolumeO2(basis; volume_flux_fv = surface_flux,
#                                                         reconstruction_mode = reconstruction_O2_full,
#                                                         slope_limiter = vanleer)
surface_flux = flux_lax_friedrichs
volume_flux = flux_central
polydeg = 2
basis = LobattoLegendreBasis(polydeg)
# Shock-capturing indicator from the non-equilibrium measure instead of the modal energy of
# the Hennemann-Gassner indicator: the element maximum of `nonequilibrium` is mapped linearly
# onto [0, alpha_max] between `lo` (pure DG below) and `hi` (alpha_max above). The blending
# then follows the physics of the moment equations: Grad13 loses its real characteristic
# speeds where the measure reaches about 0.6 (heat flux 0.64 p √θ or compressive stress
# 0.8 p in one dimension), and the shock layers of this case reach 0.4 to 0.9, while the
# expansion around the corner and the shear layer stay below 0.25 and the smooth flow below
# 0.03 at Kn = 0.001. With `lo = 0.1` and `hi = 0.3`, the shock layers get the full blending,
# their tails and the shear layer a partial one, and about a quarter of the elements are
# blended at all, compared with half for the Hennemann-Gassner indicator with
# `density_pressure`, at the same solution. The thresholds are absolute values of the
# measure; for other Knudsen numbers, the smooth-flow level scales with Kn.
struct IndicatorNonequilibriumBlending{RealT, Indicator, Cache} <: Trixi.AbstractIndicator
    alpha_max    :: RealT
    lo           :: RealT
    hi           :: RealT
    alpha_smooth :: Bool
    indicator    :: Indicator # IndicatorMax of `nonequilibrium`
    cache        :: Cache     # its `alpha` (saved as element variable) and a work array
end

function IndicatorNonequilibriumBlending(equations, basis; alpha_max = 0.5, lo = 0.05,
                                         hi = 0.2, alpha_smooth = true)
    indicator = IndicatorMax(equations, basis, variable = nonequilibrium)
    cache = (; alpha = indicator.cache.alpha, alpha_tmp = similar(indicator.cache.alpha))
    return IndicatorNonequilibriumBlending(promote(alpha_max, lo, hi)..., alpha_smooth,
                                           indicator, cache)
end

function (indicator::IndicatorNonequilibriumBlending)(u, mesh, equations, dg, cache;
                                                      kwargs...)
    alpha = indicator.indicator(u, mesh, equations, dg, cache; kwargs...)
    (; alpha_max, lo, hi) = indicator
    @. alpha = alpha_max * clamp((alpha - lo) / (hi - lo), 0, 1)
    if indicator.alpha_smooth # as in the HG indicator: neighbours get at least half the alpha
        alpha_tmp = indicator.cache.alpha_tmp
        resize!(alpha_tmp, length(alpha))
        Trixi.apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
    end
    return alpha
end

indicator_sc = IndicatorNonequilibriumBlending(equations, basis;
                                               alpha_max = 0.5, lo = 0.1, hi = 0.3)
volume_integral = VolumeIntegralShockCapturingRRG(basis, indicator_sc;
                                                  volume_flux_dg = volume_flux,
                                                  volume_flux_fv = surface_flux,
                                                  slope_limiter = vanleer,
                                                  cons2recon = cons2prim,
                                                  recon2cons = prim2cons)
solver = DGSEM(basis, surface_flux, volume_integral)

# 21 square trees of side 1/3; level 2 gives elements of size 1/12
mesh = p4est_mesh_forward_facing_step(polydeg = polydeg, initial_refinement_level = 2)

# Supersonic inflow by prescribing the full state, open outflow, specular walls
boundary_conditions = (; inflow = BoundaryConditionDirichlet(initial_condition),
                       outflow = boundary_condition_do_nothing,
                       wall = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions,
                                    source_terms = source_terms_grad13)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (max_collision_frequency,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# The solution is saved in the output variables of the finite-volume solver
save_solution = SaveSolutionCallback(interval = 500,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2moments)

# Refine where the gas is far from equilibrium: the element maximum of the stress and heat
# flux relative to the pressure. Level 6 gives elements of size 1/192 (node spacing 1/384, as
# the 1/360 cells of the finite-volume reference); the shock layers at Kn = 0.001 are a few
# 1/100 wide. The low thresholds also refine the reflected shock and the shear layer from the
# corner, which are weaker than the bow shock; with 0.02 and 0.1 they stay at the medium
# level, where the numerical dissipation smears them. The mesh grows to about 14000 elements
# at t = 1.
amr_indicator = IndicatorMax(semi, variable = nonequilibrium)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 2,
                                      med_level = 4, med_threshold = 0.01,
                                      max_level = 6, max_threshold = 0.04)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

# CFL condition of the hyperbolic part; it also has to keep the explicit relaxation stable
stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# The expansion around the corner of the step drives the gas towards vacuum; the
# positivity-preserving limiter keeps the density and the pressure (which is concave in the
# conservative variables, as for the Euler equations) of the nodal values positive by scaling
# towards the element means
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-6, 1.0e-6),
                                                     variables = (Trixi.density,
                                                                  Trixi.pressure))

sol = solve(ode, CarpenterKennedy2N54(; stage_limiter!, williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize callback
            ode_default_options()..., callback = callbacks);
