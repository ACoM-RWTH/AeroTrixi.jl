using Trixi
using AeroTrixi

###############################################################################
# Semidiscretization of the visco-resistive MHD equations

# Magnetized, compressible von Karman vortex street behind a cylinder.
# Resource for this testcase:
# - T. C. Warburton, G. E. Karniadakis (1999)
#   A Discontinuous Galerkin Method for the Viscous MHD Equations
#   Journal of Computational Physics 152(2):608-641
#   [DOI: 10.1006/jcph.1999.6248](https://doi.org/10.1006/jcph.1999.6248)

# Fluid parameters (Warburton & Karniadakis give no value for these)
gamma() = 5 / 3
prandtl_number() = 0.72

# Parameters for compressible von-Karman vortex street
Re() = 200
Ma() = 0.5f0
D() = 1 # Diameter of the cylinder as in the mesh file

# Parameters that can be freely chosen
v_in() = 1
p_in() = 1

# Parameters that follow from Reynolds and Mach number + adiabatic index gamma
c() = v_in() / Ma()

p_over_rho() = c()^2 / gamma()
rho_in() = p_in() / p_over_rho()

mu() = rho_in() * v_in() * D() / Re()

# MHD additions
Alfven_Mach_number() = 0.1
# Use definition of Alfven (Mach) number as in Warburton & Karniadakis
Alfven_speed() = v_in() * Alfven_Mach_number()
B_in() = Alfven_speed() * sqrt(rho_in())

S_v() = Alfven_Mach_number() * Re() # viscous Lundquist number
S_r() = S_v() # Use resistive = viscous Lundquist number as in Warburton & Karniadakis
eta() = D() * Alfven_speed() / S_r()

equations = IdealGlmMhdEquations2D(gamma())
equations_parabolic = ViscoResistiveMhd2D(equations, mu = mu(),
                                          Prandtl = prandtl_number(),
                                          eta = eta(),
                                          gradient_variables = GradientVariablesPrimitive())

@inline function initial_condition(x, t, equations)
    # set the freestream flow parameters
    rho = rho_in()
    v1 = v_in()
    v2 = 0.0
    v3 = 0.0
    p = p_in()

    B1 = B_in()
    B2 = 0.0
    B3 = 0.0
    psi = 0.0

    prim = SVector(rho, v1, v2, v3, p, B1, B2, B3, psi)
    return prim2cons(prim, equations)
end

# Mesh which is refined around the cylinder and the wake region
mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/7312faba9a50ef506b13f01716b4ec26/raw/f08b4610491637d80947f1f2df483c81bd2cb071/cylinder_vortex_street.inp",
                           joinpath(@__DIR__, "cylinder_vortex_street.inp"))

mesh = P4estMesh{2}(mesh_file)

bc_freestream = BoundaryConditionDirichlet(initial_condition)

# Boundary names are those assigned in HOHQMesh.jl
boundary_conditions = (; Circle = boundary_condition_slip_wall_no_magnetic_field, # top half of the cylinder
                       Circle_R = boundary_condition_slip_wall_no_magnetic_field, # bottom half of the cylinder
                       Top = bc_freestream,
                       Top_R = bc_freestream, # aka bottom
                       Right = bc_freestream,
                       Right_R = bc_freestream,
                       Left = bc_freestream,
                       Left_R = bc_freestream)

# Inflow: prescribed free-stream velocity and magnetic field
velocity_bc_free = NoSlip((x, t, equations) -> SVector(v_in(), 0.0, 0.0))
heat_bc_free = Adiabatic((x, t, equations) -> 0.0)
magnetic_bc_free = Isomagnetic((x, t, equations) -> SVector(B_in(), 0.0, 0.0))
boundary_condition_free = BoundaryConditionVRMHDWall(velocity_bc_free, heat_bc_free,
                                                     magnetic_bc_free)

# Cylinder: no-slip, adiabatic, non-magnetic wall
velocity_bc_cylinder = NoSlip((x, t, equations) -> SVector(0.0, 0.0, 0.0))
heat_bc_cylinder = Adiabatic((x, t, equations) -> 0.0)
magnetic_bc_cylinder = Isomagnetic((x, t, equations) -> SVector(0.0, 0.0, 0.0))
boundary_condition_cylinder = BoundaryConditionVRMHDWall(velocity_bc_cylinder,
                                                         heat_bc_cylinder,
                                                         magnetic_bc_cylinder)

# At the top, bottom, and outflow boundaries, the parabolic fluxes are taken from the inside
boundary_conditions_parabolic = (; Circle = boundary_condition_cylinder, # top half of the cylinder
                                 Circle_R = boundary_condition_cylinder, # bottom half of the cylinder
                                 Top = boundary_condition_do_nothing,
                                 Top_R = boundary_condition_do_nothing, # aka bottom
                                 Right = boundary_condition_do_nothing,
                                 Right_R = boundary_condition_do_nothing,
                                 Left = boundary_condition_free,
                                 Left_R = boundary_condition_free)

# Set the numerical fluxes for the volume and the surface contributions.
surface_flux = (flux_hll, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)

polydeg = 4
basis = LobattoLegendreBasis(polydeg)

# Need stabilization for VRMHD case
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

# Combine all the spatial discretization components into a high-level description.
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# Setup an ODE problem

tspan = (0.0, 120.0)
ode = semidiscretize(semi, tspan)

# Callbacks
summary_callback = SummaryCallback()

analysis_interval = 100_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = ())

alive_callback = AliveCallback(alive_interval = 200)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

# The CFL number is ramped up linearly from `cfl_0` to `cfl_max` during the transient
# start-up phase until `t_ramp_up`. The values are tuned for `Trixi.RelaxationCKL54()`
# and `polydeg = 4`.
cfl_0() = 0.5
cfl_max() = 1.8
t_ramp_up() = 3.0

cfl(t) = min(cfl_max(), cfl_0() + t / t_ramp_up() * (cfl_max() - cfl_0()))

stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        glm_speed_callback,
                        stepsize_callback,
                        save_solution)

###############################################################################
# run the simulation

# Fourth-order Carpenter-Kennedy low-storage Runge-Kutta method with entropy relaxation
ode_algorithm = Trixi.RelaxationCKL54()

sol = Trixi.solve(ode, ode_algorithm;
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()..., callback = callbacks);
