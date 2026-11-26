using OrdinaryDiffEqSSPRK
using OrdinaryDiffEqCore: PIDController
using Trixi
using AeroTrixi

###############################################################################
# Transonic flow around the ONERA OAT 15A supercritical airfoil at the flow conditions
# of the transonic buffet experiment
#   Mach number M = 0.73, angle of attack alpha = 3.5 degrees,
#   Reynolds number Re = 3 * 10^6 (based on the chord length),
# see
# - L. Jacquin, P. Molton, S. Deck, B. Maury, D. Soulevant (2009)
#   Experimental Study of Shock Oscillation over a Transonic Supercritical Profile.
#   AIAA Journal 47(9):1985-1994.
#   [DOI: 10.2514/1.30190](https://doi.org/10.2514/1.30190)
# - S. Deck (2005)
#   Numerical Simulation of Transonic Buffet over a Supercritical Airfoil.
#   AIAA Journal 43(7):1556-1566.
#   [DOI: 10.2514/1.9885](https://doi.org/10.2514/1.9885)
#
# The mesh with boundary layers is generated with cfMesh, see `utils/cfmesh` and the
# documentation page "Boundary-layer meshes with cfMesh".

gamma() = 1.4
equations = CompressibleEulerEquations2D(gamma())

airfoil_chord_length() = 1.0
Re() = 3.0e6
mach_inf() = 0.73
aoa() = deg2rad(3.5)

# Non-dimensionalization with free-stream density rho_inf = gamma and pressure p_inf = 1,
# i.e., the free-stream speed of sound is c_inf = 1 and u_inf = mach_inf.
rho_inf() = gamma()
p_inf() = 1.0
u_inf() = mach_inf()

mu() = rho_inf() * u_inf() * airfoil_chord_length() / Re()
prandtl_number() = 0.72
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

@inline function initial_condition_freestream(x, t, equations)
    v1 = u_inf() * cos(aoa())
    v2 = u_inf() * sin(aoa())

    prim = SVector(rho_inf(), v1, v2, p_inf())
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_freestream

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 1.0,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)

# In non-blended/limited regions, we use the cheaper weak form volume integral
volume_integral_default = VolumeIntegralWeakForm()

surface_flux = flux_hllc
volume_flux = flux_ranocha
# For the blended/limited regions, we need to supply high-order and low-order volume integrals.
volume_integral_blend_high_order = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_blend_low_order = VolumeIntegralPureLGLFiniteVolumeO2(basis;
                                                                      volume_flux_fv = surface_flux,
                                                                      reconstruction_mode = reconstruction_O2_inner,
                                                                      slope_limiter = minmod)

volume_integral = VolumeIntegralShockCapturingHGType(shock_indicator;
                                                     volume_integral_default = volume_integral_default,
                                                     volume_integral_blend_high_order = volume_integral_blend_high_order,
                                                     volume_integral_blend_low_order = volume_integral_blend_low_order)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

###############################################################################

# Mesh with 35368 straight-sided quadrilaterals and 20 boundary layers with growth ratio 1.2
# (first layer height ~2.5e-5 chord lengths) in a circular domain with radius 50 chord
# lengths, generated with cfMesh by `utils/cfmesh/Allmesh`. The boundaries are given as
# node sets `airfoil` and `farfield`.
mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/148f027c7f59239b9983e06e5c16f7c2/raw/d32e626490c042a6350ff692b87d038f19ae02e4/oat15a_airfoil.inp",
                           joinpath(@__DIR__, "oat15a_airfoil.inp"))

boundary_symbols = [:airfoil, :farfield]
mesh = P4estMesh{2}(mesh_file, boundary_symbols = boundary_symbols)

boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

boundary_conditions_hyp = (; farfield = boundary_condition_free_stream,
                           airfoil = boundary_condition_slip_wall)

boundary_conditions_para = (; farfield = boundary_condition_free_stream,
                            airfoil = boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions_hyp,
                                                                    boundary_conditions_para))

###############################################################################
# ODE solvers, callbacks etc.

# The experimentally observed buffet frequency corresponds to a period of roughly
# 15 convective time units.
t_c = airfoil_chord_length() / u_inf() # convective time
tspan = (0.0, 50 * t_c)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

force_boundary_names = (:airfoil,)
drag_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           DragCoefficientPressure2D(aoa(), rho_inf(),
                                                                     u_inf(),
                                                                     airfoil_chord_length()))
lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure2D(aoa(), rho_inf(),
                                                                     u_inf(),
                                                                     airfoil_chord_length()))
drag_coefficient_shear_force = AnalysisSurfaceIntegral(force_boundary_names,
                                                       DragCoefficientShearStress2D(aoa(),
                                                                                    rho_inf(),
                                                                                    u_inf(),
                                                                                    airfoil_chord_length()))
lift_coefficient_shear_force = AnalysisSurfaceIntegral(force_boundary_names,
                                                       LiftCoefficientShearStress2D(aoa(),
                                                                                    rho_inf(),
                                                                                    u_inf(),
                                                                                    airfoil_chord_length()))

analysis_interval = 10_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient,
                                                           drag_coefficient_shear_force,
                                                           lift_coefficient_shear_force))

###############################################################################
# AeroTrixi addition

# Save the pressure and friction coefficients at all surface nodes of the airfoil,
# e.g., to compare the pressure distribution with the experimental data of Jacquin et al.
pressure_coefficient = SurfacePointwise(force_boundary_names,
                                        SurfacePressureCoefficient(p_inf(), rho_inf(),
                                                                   u_inf()))
friction_coefficient = SurfacePointwise(force_boundary_names,
                                        SurfaceFrictionCoefficient(rho_inf(), u_inf()))

surface_pointwise_callback = SurfacePointwiseCallback(semi, pressure_coefficient,
                                                      friction_coefficient;
                                                      interval = analysis_interval)

###############################################################################

alive_callback = AliveCallback(alive_interval = 500)

save_sol_interval = 50_000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, surface_pointwise_callback, alive_callback,
                        save_solution, save_restart)

###############################################################################
# run the simulation

ode_algorithm = SSPRK43(thread = Trixi.Threaded())

time_int_tol = 1e-4
sol = solve(ode, ode_algorithm;
            abstol = time_int_tol, reltol = time_int_tol, dt = 1e-6,
            maxiters = Inf, # long simulation
            controller = PIDController(0.55, -0.27, 0.05), # optimized for SSPRK43
            ode_default_options()..., callback = callbacks)
