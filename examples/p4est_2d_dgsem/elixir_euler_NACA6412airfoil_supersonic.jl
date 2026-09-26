using Trixi
using OrdinaryDiffEqSSPRK
using Gmsh: gmsh
using Plots

###############################################################################
# Supersonic flow (Mach 2) around a NACA 6412 airfoil.
#
# This example walks through the complete workflow of a simulation:
#   1. the equations and the initial condition
#   2. the boundary conditions
#   3. the mesh, generated from a Gmsh geometry file
#   4. the discretization in space (DG method)
#   5. the time integration with a few callbacks
#   6. a plot of the result
#
# Try changing `polydeg` and `mesh_size` below and compare the results.

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

polydeg = 2      # polynomial degree of the solution in each cell
mesh_size = 0.1  # relative size of the cells (lower -> finer mesh)

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
# 5. time integration

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

callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

# The strong shocks can make the density or pressure negative for a moment,
# which the positivity limiter prevents
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-7, 1.0e-6),
                                                     variables = (pressure,
                                                                  Trixi.density))

sol = solve(ode, SSPRK33(stage_limiter! = stage_limiter!);
            dt = 1.0, # overwritten by the `stepsize_callback`
            save_everystep = false, callback = callbacks);

###############################################################################
# 6. visualization

# Other variables can be plotted with, e.g., `plot(pd["v1"])` or `plot(pd["p"])`
pd = PlotData2D(sol)
plot(pd["rho"], size = (1300, 550))
plot!(getmesh(pd))
savefig(joinpath("out", "density.png"))
