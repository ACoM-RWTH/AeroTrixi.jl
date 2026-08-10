using OrdinaryDiffEqSSPRK
using Trixi
using AeroTrixi

###############################################################################
# semidiscretization of the compressible Euler equations

gamma() = 1.4
equations = CompressibleEulerEquations2D(gamma())

p_inf() = 1.0
rho_inf() = gamma() # Gives unit speed of sound c_inf = 1.0
mach_inf() = 0.8
aoa() = deg2rad(1.25) # 1.25 Degree angle of attack

@inline function initial_condition_mach08_flow(x, t,
                                               equations::CompressibleEulerEquations2D)
    v1 = 0.7998096216639273   # 0.8 * cos(aoa())
    v2 = 0.017451908027648896 # 0.8 * sin(aoa())

    prim = SVector(1.4, v1, v2, 1.0)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach08_flow

surface_flux = flux_lax_friedrichs
volume_flux = flux_chandrashekar

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/80af941348b0d465c9f866c8c50ea767/raw/e03fc4769ab1956c610d7d78779362a36b796e68/NACA0012_ref2_quadr.inp",
                           joinpath(@__DIR__, "NACA0012_ref2_quadr.inp"))

boundary_symbols = [:Airfoil, :Inflow, :Outflow]
mesh = P4estMesh{2}(mesh_file, boundary_symbols = boundary_symbols)

bc_farfield = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = (Inflow = bc_farfield,
                       Outflow = bc_farfield,
                       Airfoil = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################

# Run for a long time to reach a steady state
tspan = (0.0, 100.0) # 100 suffices for this mesh for stationary shock position
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

alive_callback = AliveCallback(alive_interval = 1000)

save_sol_interval = 50_000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true)

l_inf = 1.0 # Length of airfoil
force_boundary_names = (:Airfoil,)
u_inf() = mach_inf()
drag_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           DragCoefficientPressure2D(aoa(), rho_inf(),
                                                                     u_inf(), l_inf))

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure2D(aoa(), rho_inf(),
                                                                     u_inf(), l_inf))

###############################################################################
# AeroTrixi addition

pressure_coefficient = AnalysisSurfacePointwise(force_boundary_names,
                                                SurfacePressureCoefficient(p_inf(),
                                                                           rho_inf(),
                                                                           u_inf()))

analysis_interval = 500_000 # Only at the end
analysis_callback = AeroTrixi.AnalysisCallback(semi, interval = analysis_interval,
                                               output_directory = "out",
                                               analysis_errors = Symbol[],
                                               save_analysis = true,
                                               analysis_integrals = (drag_coefficient,
                                                                     lift_coefficient),
                                               analysis_pointwise = (pressure_coefficient,))

###############################################################################

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        save_solution,
                        save_restart)

###############################################################################
# run the simulation

ode_alg = SSPRK43(thread = Trixi.Threaded())

sol = solve(ode, ode_alg, dt = 1e-3, adaptive = true,
            save_everystep = false, callback = callbacks);
