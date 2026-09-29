using Trixi
using AeroTrixi
using LinearAlgebra: norm
using Downloads

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

@inline function initial_condition(x, t, equations::CompressibleEulerEquations3D)
    # set the freestream flow parameters
    rho_freestream = 1.4

    # v_total = 0.84 = Mach

    # AoA = 3.06
    v1 = 0.8388023121403883
    v2 = 0.0448406193973588
    v3 = 0.0

    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

# Ensure that rho and p are the same across symmetry line and allow only
# tangential velocity
@inline function bc_symmetry(u_inner, normal_direction::AbstractVector, x, t,
                             surface_flux_function,
                             equations::CompressibleEulerEquations3D)
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute the primitive variables
    rho, v1, v2, v3, p = cons2prim(u_inner, equations)

    v_normal = normal[1] * v1 + normal[2] * v2 + normal[3] * v3

    u_mirror = prim2cons(SVector(rho,
                                 v1 - 2 * v_normal * normal[1],
                                 v2 - 2 * v_normal * normal[2],
                                 v3 - 2 * v_normal * normal[3],
                                 p), equations)

    flux = surface_flux_function(u_inner, u_mirror, normal, equations) * norm_

    return flux
end

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

# Use Hennemann-Gassner shock indicator to identify cells which need the stabilized volume integral.
# The only relevant parameter in this case is `alpha_min`, which governs the sensitivity,
# i.e., the minimum shock indicator value at which the stabilized volume integral becomes active.
# `alpha_max` is not relevant for the volume integral, as there is no blending of fluxes performed.
# However, for the AMR callback below, `alpha_max` is relevant.
indicator = IndicatorHennemannGassner(equations, basis,
                                      alpha_max = 0.5,
                                      alpha_min = 0.01,
                                      alpha_smooth = true,
                                      variable = density_pressure)

volume_integral_weakform = VolumeIntegralWeakForm()

volume_flux = flux_ranocha
volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(volume_flux)

# Adaptive volume integral using the Hennemann-Gassner shock indicator to perform the
# stabilized/EC volume integral when needed.
volume_integral = VolumeIntegralAdaptive(indicator = indicator,
                                         volume_integral_default = volume_integral_weakform,
                                         volume_integral_stabilized = volume_integral_fluxdiff)

surface_flux = flux_hllc
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

mesh_file = Trixi.download("https://raw.githubusercontent.com/DanielDoehring/paper-2026-vta/refs/heads/main/5_NumericalExamples/5_1_WF-FD/5_1_4_ONERA_M6/m6wing_sanitized.inp",
                           joinpath(@__DIR__, "m6wing_sanitized.inp"))

boundary_symbols = [:Symmetry, :FarField, :BottomWing, :TopWing]

mesh = P4estMesh{3}(mesh_file, boundary_symbols = boundary_symbols)

boundary_conditions = (Symmetry = bc_symmetry, # Symmetry: bc_symmetry
                       FarField = bc_farfield, # Farfield: bc_farfield
                       BottomWing = boundary_condition_slip_wall, # Wing: bc_slip_wall
                       TopWing = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

#tspan = (0.0, 6.049)
#ode = semidiscretize(semi, tspan)

restart_file = "restart_tc_6049.h5"
Downloads.download("https://zenodo.org/records/18921762/files/restart_tc_6049.h5?download=1",
                   joinpath(@__DIR__, restart_file))

restart_filename = joinpath(@__DIR__, restart_file)
tspan = (load_time(restart_filename), 6.05)

# Need to overload this due to breaking change after restart files have been constructed
function Trixi.varnames(::typeof(cons2cons), ::CompressibleEulerEquations3D)
    return ("rho", "rho_v1", "rho_v2", "rho_v3", "rho_e")
end
ode = semidiscretize(semi, tspan, restart_filename)

# Callbacks
###############################################################################

summary_callback = SummaryCallback()

force_boundary_names = (:BottomWing, :TopWing)

aoa() = deg2rad(3.06)

rho_inf() = 1.4
u_inf(equations) = 0.84
# Area calculated from information given at https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html

height_ref = 1.1963
height = 1.0 # Mesh we use normalizes wing height to one

g_I = tan(deg2rad(30)) * height

#base = 0.8059
base = 0.8059 / height_ref # Mesh we use normalizes wing height to one

g_II = base - g_I
g_III = tan(deg2rad(15.8)) * height
A = height * (0.5 * (g_I + g_III) + g_II)

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure3D(aoa(), rho_inf(),
                                                                     u_inf(equations), A))

###############################################################################
# AeroTrixi additions

p_inf() = 1.0
pressure_coefficient = AnalysisSurfacePointwise(force_boundary_names,
                                                SurfacePressureCoefficient(p_inf(),
                                                                           rho_inf(),
                                                                           u_inf(equations)))

analysis_interval = 10_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (lift_coefficient,),
                                     save_analysis = true,
                                     output_directory = "out/")

# save the pressure coefficient at all surface nodes of the wing
surface_pointwise_callback = SurfacePointwiseCallback(semi, pressure_coefficient;
                                                      interval = analysis_interval)

###############################################################################

alive_callback = AliveCallback(alive_interval = 5)

save_sol_interval = analysis_interval

save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out/")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory = "out/")

## 6.049 -> 6.05 ##

Trixi.download("https://gist.githubusercontent.com/DanielDoehring/c09dbb7d86aa58fa7edb6aaa9f22a187/raw/4f5361e62125f5e8dd526f29fb13221a6ecce250/ONERA_M6_a_12_p4.txt",
               joinpath(@__DIR__, "a_12.txt"))

cd(dirname(@__FILE__))
path = "./"

ode_alg = Trixi.PairedExplicitRK4(12, path)
cfl = 10.0

stepsize_callback = StepsizeCallback(cfl = cfl, interval = 2)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        surface_pointwise_callback,
                        save_solution,
                        save_restart,
                        stepsize_callback)

# Run the simulation
###############################################################################

sol = Trixi.solve(ode, ode_alg, dt = 42.0, save_start = false,
                  save_everystep = false, callback = callbacks);
