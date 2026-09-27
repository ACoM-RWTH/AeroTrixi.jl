using Trixi
using AeroTrixi: @RunInfo
using OrdinaryDiffEqSSPRK
using Gmsh: gmsh
using Plots

###############################################################################
# Supersonic flow (Mach 2) around a NACA 6412 airfoil, with adaptive mesh refinement
# (AMR) or on a static mesh.
#
# This is `elixir_euler_NACA6412airfoil_supersonic.jl` with one addition: with AMR,
# the mesh from the .geo file is only a coarse skeleton, which is refined
# automatically where shocks are and coarsened again where they have passed. The
# additions are marked with "AMR" below.
#
# Choose a setup in section 0, run this file, and repeat for other setups. Each run
# is stored in its own directory `out/<name of this file>/<parameters>/`. Afterwards,
# `elixir_euler_NACA6412airfoil_supersonic_compare.jl` compares all stored runs.
#
# The file walks through the complete workflow of a simulation:
#   0. the choice of the setup
#   1. the equations and the initial condition
#   2. the boundary conditions
#   3. the mesh, generated from a Gmsh geometry file
#   4. the discretization in space (DG method)
#   5. the adaptive mesh refinement (AMR)
#   6. the time integration with a few callbacks
#   7. a plot of the result

###############################################################################
# 0. setup: comment in exactly one of the following lines

#! format: off
# AMR with 3 refinement levels on the coarse mesh (about 1 minute on 4 threads)
use_amr = true; mesh_size = 0.2; max_level = 3
# AMR with 4 refinement levels (about 3 minutes)
# use_amr = true; mesh_size = 0.2; max_level = 4
# static mesh with about as many cells as AMR with 3 levels at the end (about 30 s)
# use_amr = false; mesh_size = 0.065
# static mesh with about the same run time as AMR with 3 levels
# use_amr = false; mesh_size = 0.042
# static mesh with somewhat less run time than AMR with 4 levels
# use_amr = false; mesh_size = 0.034
#! format: on

polydeg = 2 # polynomial degree of the solution in each cell

###############################################################################
# 1. equations and initial condition

# compressible Euler equations of gas dynamics with ratio of specific heats 1.4
equations = CompressibleEulerEquations2D(1.4)

# freestream state: density, velocity in x and y, pressure.
# The speed of sound is sqrt(1.4 * 1.0 / 1.4) = 1, so v1 = 2 means Mach 2.
@inline function initial_condition_mach2_flow(x, t,
                                              equations::CompressibleEulerEquations2D)
    rho_freestream = 1.4
    v1 = 2.0
    v2 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations) # convert primitive to conservative variables
end

initial_condition = initial_condition_mach2_flow

###############################################################################
# 2. boundary conditions

# At a supersonic inflow all information enters the domain,
# so the flux is computed from the freestream state alone.
@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t, surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_mach2_flow(x, t, equations)
    return flux(u_boundary, normal_direction, equations)
end

# At a supersonic outflow all information leaves the domain,
# so the flux is computed from the state inside the domain alone.
@inline function boundary_condition_supersonic_outflow(u_inner,
                                                       normal_direction::AbstractVector,
                                                       x, t, surface_flux_function,
                                                       equations::CompressibleEulerEquations2D)
    return flux(u_inner, normal_direction, equations)
end

###############################################################################
# 3. mesh

# Gmsh reads the geometry file and generates a mesh of quadrilaterals,
# which is written in the Abaqus (.inp) format that Trixi.jl can read
geo_file = joinpath(@__DIR__, "NACA6412airfoil.geo")
mkpath("out")
mesh_file = joinpath("out", "NACA6412airfoil.inp")

# same as `gmsh -setnumber meshSize 0.1` on the command line:
# sets the parameter `meshSize` used in the .geo file
gmsh.initialize(["gmsh", "-setnumber", "meshSize", string(mesh_size)])
gmsh.option.setNumber("General.Terminal", 0)      # do not print Gmsh's log
gmsh.open(geo_file)
gmsh.model.mesh.generate(2)                       # 2D mesh
gmsh.write(mesh_file)
gmsh.finalize()

# The boundaries are the named `Physical Line`s of the .geo file
mesh = P4estMesh{2}(mesh_file, polydeg = polydeg,
                    boundary_symbols = [:inflow, :outflow, :airfoil, :walls])

boundary_conditions = (; inflow = boundary_condition_supersonic_inflow,
                       outflow = boundary_condition_supersonic_outflow,
                       airfoil = boundary_condition_slip_wall,
                       walls = boundary_condition_slip_wall)

###############################################################################
# 4. spatial discretization with the discontinuous Galerkin spectral element method

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

# The shock indicator detects cells containing shocks, where the high-order DG
# method is blended with a robust low-order finite volume method
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

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# 5. AMR: adaptive mesh refinement (only if `use_amr = true`)

if use_amr
    # The indicator measures how strongly the density varies within each cell;
    # it is large at shocks
    amr_indicator = IndicatorLoehner(semi, variable = Trixi.density)

    # Cells with an indicator above `med_threshold` are refined to `med_level`,
    # cells above `max_threshold` to `max_level`, and all others are coarsened back
    # to `base_level`, i.e., the initial mesh. Each level halves the cell size, so
    # with `max_level = 3` the cells at the shocks are 8 times smaller than initially.
    amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                          base_level = 0,
                                          med_level = 2, med_threshold = 0.02,
                                          max_level = max_level, max_threshold = 0.05)

    # adapt the mesh every 10 time steps
    amr_callback = AMRCallback(semi, amr_controller, interval = 10)
end

###############################################################################
# 6. time integration

tspan = (0.0, 0.6)
ode = semidiscretize(semi, tspan)

# print a summary of the setup at the start and timings at the end
summary_callback = SummaryCallback()

# print some information about the solution every 100 time steps.
# There is no exact solution to compare with, so no errors are computed.
# TODO: Trixi.jl and AeroTrixi.jl both export an `AnalysisCallback`; this is the one
# of Trixi.jl, which is only unambiguous as long as `using AeroTrixi` is not called.
# Resolve the name clash in AeroTrixi.jl and update this example accordingly.
analysis_callback = AnalysisCallback(semi, interval = 100, analysis_errors = Symbol[])

# choose the time step from the CFL condition
stepsize_callback = StepsizeCallback(cfl = 0.5)

# Store the results of this run in `out/<name of this file>/<parameters>/` together
# with a description of the run (`run.toml`), a copy of this file and of the .geo file,
# and the final solution. Running the same setup again replaces these results.
if use_amr
    parameters = (; use_amr, polydeg, mesh_size, max_level)
else
    parameters = (; use_amr, polydeg, mesh_size)
end
run_info = @RunInfo(semi; parameters, files = [geo_file])

# the AMR callback has to come before the `stepsize_callback`, so that the
# time step is computed on the adapted mesh
if use_amr
    callbacks = CallbackSet(summary_callback, analysis_callback, amr_callback,
                            stepsize_callback, run_info.callback)
else
    callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback,
                            run_info.callback)
end

# The strong shocks can make the density or pressure negative for a moment,
# which the positivity limiter prevents
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-7, 1.0e-6),
                                                     variables = (pressure,
                                                                  Trixi.density))

sol = solve(ode, SSPRK33(stage_limiter! = stage_limiter!);
            dt = 1.0, # overwritten by the `stepsize_callback`
            save_everystep = false, callback = callbacks);

###############################################################################
# 7. visualization

# Other variables can be plotted with, e.g., `plot(pd["v1"])` or `plot(pd["p"])`
pd = PlotData2D(sol)
plot(pd["rho"], size = (1300, 550))
plot!(getmesh(pd))
savefig(joinpath(run_info.dir, "density.png"))
