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
# frequency ν = ρ √θ / Kn, so the time step is limited by ν Δt ≲ 1 in addition to the CFL
# condition; the `RelaxationStepsizeCallback` enforces this. At Kn = 0.001 this limit is
# dominant on the coarse base level, at Kn = 0.005 both limits are comparable.

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

# Supersonic inflow by prescribing the full state, open outflow, specular walls
boundary_conditions = (; inflow = BoundaryConditionDirichlet(initial_condition),
                       outflow = boundary_condition_do_nothing,
                       wall = boundary_condition_slip_wall)

# Lax-Friedrichs flux with the crude wave-speed bounds of the finite-volume solver. The shock
# in front of the step is a reflected shock with a relative Mach number of 3, far beyond the
# critical Mach number of Grad13, so that the moment equations are not hyperbolic inside the
# shock layer. With the less dissipative HLL flux (`flux_hll_grad13`), a density spike grows at
# the head of the shock and the run breaks down; the additional dissipation of the
# Lax-Friedrichs flux keeps the shock layer at the values of the finite-volume solution. There
# is no entropy-conservative two-point flux for Grad13, so the DG volume integral uses the
# central flux (standard DGSEM) and is blended with the subcell finite-volume scheme in
# troubled elements.
surface_flux = flux_lax_friedrichs
volume_flux = flux_central
polydeg = 2
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# 21 square trees of side 1/3; level 2 gives elements of size 1/12
mesh = p4est_mesh_forward_facing_step(polydeg = polydeg, initial_refinement_level = 2)

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
# flux relative to the pressure. In addition, the elements within the distance `step_distance`
# of the face and the corner of the step are always refined to the maximum level: the shocks
# are born at the face of the step, where the inflow is stopped within a few time steps, and
# the corner expansion is singular. Refining an element only after such a front has formed inside it projects the
# front onto children elements with negative mean pressure. Level 5 gives elements of size 1/96.
"""
    IndicatorNonequilibriumStep(semi; step_distance, step_value)

AMR indicator of the forward-facing step: the `IndicatorMax` of [`nonequilibrium`](@ref), raised
to at least `step_value` in the elements whose center lies within `step_distance` of the face or
the corner of the step.
"""
struct IndicatorNonequilibriumStep{Indicator, RealT, Cache} <: Trixi.AbstractIndicator
    indicator     :: Indicator
    step_distance :: RealT
    step_value    :: RealT
    cache         :: Cache # the `alpha` of `indicator`, saved as element variable `indicator_amr`
end

function IndicatorNonequilibriumStep(semi; step_distance, step_value)
    indicator = IndicatorMax(semi, variable = nonequilibrium)
    return IndicatorNonequilibriumStep(indicator, step_distance, step_value,
                                       (; alpha = indicator.cache.alpha))
end

# distance of `x` from the face of the step (the segment x = 1, 0 ≤ y ≤ 1/3) and from its
# corner (1, 1/3)
function distance_to_step(x)
    if x[1] <= 1
        return sqrt((1 - x[1])^2 + max(x[2] - 1 / 3, 0)^2)
    else
        return sqrt((x[1] - 1)^2 + (x[2] - 1 / 3)^2)
    end
end

function (indicator::IndicatorNonequilibriumStep)(u, mesh, equations, dg, cache; kwargs...)
    alpha = indicator.indicator(u, mesh, equations, dg, cache; kwargs...)
    node_coordinates = cache.elements.node_coordinates
    n = nnodes(dg)
    for element in eachelement(dg, cache)
        # element center as the mean of its corner nodes
        center = (Trixi.get_node_coords(node_coordinates, equations, dg, 1, 1, element) +
                  Trixi.get_node_coords(node_coordinates, equations, dg, n, 1, element) +
                  Trixi.get_node_coords(node_coordinates, equations, dg, 1, n, element) +
                  Trixi.get_node_coords(node_coordinates, equations, dg, n, n, element)) / 4
        if distance_to_step(center) <= indicator.step_distance
            alpha[element] = max(alpha[element], indicator.step_value)
        end
    end
    return alpha
end

amr_indicator = IndicatorNonequilibriumStep(semi; step_distance = 0.1, step_value = 1.0)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 2,
                                      med_level = 4, med_threshold = 0.05,
                                      max_level = 5, max_threshold = 0.2)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

# CFL condition of the hyperbolic part, then the stability limit of the explicit relaxation
stepsize_callback = StepsizeCallback(cfl = 0.5)
relaxation_stepsize_callback = RelaxationStepsizeCallback(nu_dt_max = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback,
                        save_solution,
                        stepsize_callback, relaxation_stepsize_callback)

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
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize callbacks
            ode_default_options()..., callback = callbacks);
