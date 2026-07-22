using OrdinaryDiffEqSSPRK
using Trixi
using AeroTrixi
using FlowRef

###############################################################################
# Hypersonic O2 flow over a cylinder, one species, thermal equilibrium.
#
# Only a quarter of the cylinder is meshed: the radial direction runs from the
# cylinder surface out to a shock-fitted outer boundary, the angular direction
# from the stagnation line (y = 0) to the lateral outflow plane (x = 0).

# free-stream conditions
const MASS = 5.3134e-26         # O2 molecular mass, kg
const Θ_VIBR = 2273.54          # O2 characteristic vibrational temperature, K

const V1_FREESTREAM = 5956.0    # m / s
const V2_FREESTREAM = 0.0
const P_FREESTREAM = 476.0      # Pa
const T_FREESTREAM = 901.0      # K
const L_REF = 0.045             # cylinder radius, m

const R_SPECIFIC = FlowRef.k_B / MASS
const RHO_FREESTREAM = P_FREESTREAM / (R_SPECIFIC * T_FREESTREAM)

# every reference quantity follows from (p, T, rho, L); the equations are solved
# in these units, so the free stream is O(1) by construction
const REF_Q = p_T_rho_L(P_FREESTREAM, T_FREESTREAM, RHO_FREESTREAM, L_REF)

###############################################################################
# mesh

# `xi_` runs from the outer (shock) boundary at -1 to the cylinder surface at +1,
# `eta_` from the stagnation line at -1 to the lateral outflow plane at +1
function mapping_full(xi_, eta_, cyl_radius, points_shock)
    shock_pos = [(points_shock[1], 0.0), (points_shock[2], points_shock[2]),
                 (0.0, points_shock[3])]  # 3 points that define shock

    # spline has form R[1] + c * eta_01^2 + d * eta_01^3, derivative w.r.t eta_01 is 0 at eta_01 = 0
    R = [sqrt(shock_pos[i][1]^2 + shock_pos[i][2]^2) for i in 1:3]  # 3 radii
    spline_matrix = [1.0 1.0; 0.25 0.125]  # find cubic spline coefficients
    spline_RHS = [R[3] - R[1], R[2] - R[1]]
    spline_cd = spline_matrix \ spline_RHS

    eta_01 = (eta_ + 1) / 2
    R_outer = R[1] + spline_cd[1] * eta_01^2 + spline_cd[2] * eta_01^3
    angle = -π / 4 + eta_ * π / 4

    xi_01 = 0.5 * (-xi_ + 1.0)

    r = (cyl_radius + xi_01 * (R_outer - cyl_radius))

    return SVector(round(r * sin(angle); digits = 8), round(r * cos(angle); digits = 8))
end

Nx = 30 # or 60
Ny = 30
polydeg = 3

if Nx == 30
    points_shock = [1.2947, 1.0127, 2.163]
    cyl_radius = 1.0
else
    points_shock = [1.294, 1.0115, 2.162]
    cyl_radius = 1.0
end

mapping = (xi_, eta_) -> mapping_full(xi_, eta_, cyl_radius, points_shock)

trees_per_dimension = (Nx, Ny)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = polydeg, initial_refinement_level = 0,
                 mapping = mapping,
                 periodicity = (false, false))

###############################################################################
# equations

e_int_f = T -> e_rot_cont(MASS, T) + e_vibr_iho(MASS, Θ_VIBR, T)
c_int_f = T -> c_rot_cont(MASS, T) + c_vibr_iho(MASS, Θ_VIBR, T)

equations = CompressibleEulerEquationsMs1T2D(REF_Q, [MASS], [e_int_f], [c_int_f];
                                             T_min = 10.0, T_max = 4.0e4, ΔT = 1.0,
                                             min_T_jump = 1e-5)

###############################################################################
# initial and boundary conditions

# primitive variables are (v1, v2, T, rho_1, ..., rho_NCOMP)
@inline function initial_condition_supersonic_flow(x, t,
                                                   equations::CompressibleEulerEquationsMs1T2D)
    prim = SVector(V1_FREESTREAM / REF_Q.v_ref,
                   V2_FREESTREAM / REF_Q.v_ref,
                   T_FREESTREAM / REF_Q.T_ref,
                   RHO_FREESTREAM / REF_Q.rho_ref)
    return prim2cons(prim, equations)
end

@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t, surface_flux_function,
                                                      equations::CompressibleEulerEquationsMs1T2D)
    u_boundary = initial_condition_supersonic_flow(x, t, equations)
    return flux(u_boundary, normal_direction, equations)
end

# supersonic outflow: the boundary flux is the interior flux
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector,
                                            x, t, surface_flux_function,
                                            equations::CompressibleEulerEquationsMs1T2D)
    return flux(u_inner, normal_direction, equations)
end

initial_condition = initial_condition_supersonic_flow

boundary_conditions = (; x_neg = boundary_condition_supersonic_inflow, # outer/shock
                       x_pos = boundary_condition_slip_wall,           # cylinder surface
                       y_neg = boundary_condition_slip_wall,           # symmetry, y = 0
                       y_pos = boundary_condition_outflow)             # lateral outflow

###############################################################################
# solver

surface_flux = FluxLaxFriedrichs(max_abs_speed)
volume_flux = flux_oblapenko

basis = LobattoLegendreBasis(polydeg)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# output in dimensional units

function cons2prim_scaled(u, equations::CompressibleEulerEquationsMs1T2D)
    _, _, index_c, fracpos_c, prim = AeroTrixi.cons2prim_with_index(u, equations)
    v1, v2, T = prim[1], prim[2], prim[3]

    rho = density(u, equations)
    rho_inv = 1 / rho
    p = pressure(T, u, equations)

    gamma_T = AeroTrixi.get_gamma(u, rho_inv, index_c, fracpos_c, equations.thermodata)
    a = sqrt(gamma_T * p * rho_inv)

    return SVector(rho * REF_Q.rho_ref, v1 * REF_Q.v_ref, v2 * REF_Q.v_ref,
                   p * REF_Q.p_ref, sqrt(v1^2 + v2^2) / a, T * REF_Q.T_ref, gamma_T)
end

function Trixi.varnames(::typeof(cons2prim_scaled),
                        ::CompressibleEulerEquationsMs1T2D)
    return ("rho", "v1", "v2", "p", "M", "T", "gamma")
end

###############################################################################
# ODE and callbacks
tmax = 0.1

tspan = (0.0, tmax)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

alive_callback = AliveCallback(alive_interval = 100)
analysis_interval = 100
# qualified because both Trixi and AeroTrixi export an `AnalysisCallback`; AeroTrixi's
# adds surface pointwise analysis, which is not used here
analysis_callback = Trixi.AnalysisCallback(semi, interval = analysis_interval)
stepsize_callback = StepsizeCallback(cfl = 0.7)

save_solution = SaveSolutionCallback(dt = 0.1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim_scaled)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        save_solution)

sol = solve(ode, SSPRK932();
            dt = 1.0, # overwritten by the stepsize_callback
            adaptive = false, # SSPRK932 is adaptive by default, which the callback forbids
            maxiters = 9999999, ode_default_options()...,
            callback = callbacks);
