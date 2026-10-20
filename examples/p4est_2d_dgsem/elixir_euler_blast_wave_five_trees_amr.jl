using OrdinaryDiffEqLowStorageRK
using Trixi
using AeroTrixi
using Plots

###############################################################################
# Circular blast wave on the five-tree mesh of the box [-1, 1]^2.
#
# The mesh consists of the square [-l, l]^2 in the middle and four trapezoids connecting
# it to the sides of the box, each split into `trees_per_block × trees_per_block` trees.
# The blast wave crosses tree interfaces with differing face numbering and orientation,
# vertices shared by three and four trees, and non-affine elements, while AMR creates
# hanging faces on all of them.
#
# Besides AeroTrixi.jl and Trixi.jl, this example requires the packages
# OrdinaryDiffEqLowStorageRK.jl and Plots.jl in the active environment.

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A medium blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables
    rho = r > 0.5 ? 1.0 : 1.1691
    v1 = r > 0.5 ? 0.0 : 0.1882 * cos_phi
    v2 = r > 0.5 ? 0.0 : 0.1882 * sin_phi
    p = r > 0.5 ? 1.0E-3 : 1.245

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# Five blocks with 2 × 2 trees each and a center square of half-width l = 0.4
mesh = p4est_mesh_five_trees(l = 0.4, trees_per_block = 2, polydeg = 3,
                             initial_refinement_level = 1)

# Open boundaries: the outer state is taken from the interior
boundary_conditions = (; x_neg = boundary_condition_do_nothing,
                       x_pos = boundary_condition_do_nothing,
                       y_neg = boundary_condition_do_nothing,
                       y_pos = boundary_condition_do_nothing)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 0.5,
                                          alpha_min = 0.001,
                                          alpha_smooth = true,
                                          variable = density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 1,
                                      med_level = 3, med_threshold = 0.05,
                                      max_level = 4, max_threshold = 0.2)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);

###############################################################################
# plot the final density together with the adapted mesh

pd = PlotData2D(sol)
plot_density = plot(pd["rho"], title = "density at t = $(sol.t[end])")
plot!(plot_density, getmesh(pd), linecolor = :white, linewidth = 0.3)
plot_mesh = plot(getmesh(pd), linewidth = 0.5, title = "mesh ($(Trixi.ncells(mesh)) cells)",
                 aspect_ratio = 1, xlims = (-1, 1), ylims = (-1, 1))
plot(plot_density, plot_mesh, layout = (1, 2), size = (1100, 450))
savefig(joinpath("out", "blast_wave_five_trees.png"))
