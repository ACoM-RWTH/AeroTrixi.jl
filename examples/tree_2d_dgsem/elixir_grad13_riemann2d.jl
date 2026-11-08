using OrdinaryDiffEqLowStorageRK
using Trixi
using AeroTrixi

###############################################################################
# Two-dimensional Riemann problem for Grad's 13-moment equations: a gas at rest with uniform
# temperature and a circular region of five-fold density and pressure, which expands into the
# surrounding gas with a circular shock, a contact, and a rarefaction. This is the standard test
# case `cases/square4.txt` of the finite-volume solver `MultiBlockMomentsFV` (M. Torrilhon),
# whose reference solution `reference/square4.ref` serves as a validation of the DG
# discretization: the domain [-1, 1]² is open at the left and right and bounded by specular
# walls at the bottom and top.
#
# The collision terms are explicit source terms; the `RelaxationStepsizeCallback` keeps
# ν Δt bounded, which is the dominant time step restriction at Kn = 0.001 on this mesh.

Kn = 0.001
equations = Grad13Equations2D(Kn)

"""
    initial_condition_riemann2d(x, t, equations::Grad13Equations2D)

Gas at rest in equilibrium with temperature 1 and density 5 inside the circle of radius 0.4
around the origin and density 1 outside.
"""
function initial_condition_riemann2d(x, t, equations::Grad13Equations2D)
    r = sqrt(x[1]^2 + x[2]^2)
    rho = r < 0.4 ? 5.0 : 1.0
    theta = 1.0
    return prim2cons(SVector(rho, 0.0, 0.0, theta, theta, 0.0, theta, 0.0, 0.0), equations)
end
initial_condition = initial_condition_riemann2d

boundary_conditions = (; x_neg = boundary_condition_do_nothing,
                       x_pos = boundary_condition_do_nothing,
                       y_neg = boundary_condition_slip_wall,
                       y_pos = boundary_condition_slip_wall)

surface_flux = flux_hll_grad13
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

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions,
                                    source_terms = source_terms_grad13)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (max_collision_frequency,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 200,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2moments)

stepsize_callback = StepsizeCallback(cfl = 0.5)
relaxation_stepsize_callback = RelaxationStepsizeCallback(nu_dt_max = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback, relaxation_stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize callbacks
            ode_default_options()..., callback = callbacks);
