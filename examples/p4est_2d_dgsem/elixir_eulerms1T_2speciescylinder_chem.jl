using OrdinaryDiffEqSSPRK
using Trixi
using AeroTrixi
using FlowRef

###############################################################################
# Hypersonic O2/O flow over a cylinder, two species in thermal equilibrium, with
# dissociation O2 + M -> 2O + M at Arrhenius rates.
#
# Only a quarter of the cylinder is meshed: the radial direction runs from the
# cylinder surface out to a shock-fitted outer boundary, the angular direction
# from the stagnation line (y = 0) to the lateral outflow plane (x = 0).
#
#                            |
#                            y_pos
#                            |
#                            |
#                            .
#                          .
#                        .  <- x_neg
#                      .
#  _______y_neg_______.

# free-stream conditions
const MASS_MOL = 5.3134e-26     # O2 molecular mass, kg
const MASS_ATOM = MASS_MOL / 2  # O atomic mass, kg

const Θ_VIBR = 2273.54          # characteristic vibrational temperature, K
const Θ_VIBR_ANH = 17.366       # anharmonicity, K
const E_DISS = 59364.8          # O2 dissociation energy, K
const E_FORM_O2 = 0.0           # formation energies, K
const E_FORM_O = 0.5 * E_DISS

const V1_FREESTREAM = 4000.0    # m / s
const V2_FREESTREAM = 0.0
const P_FREESTREAM = 500.0      # Pa
const T_FREESTREAM = 400.0      # K
const L_REF = 0.045             # cylinder radius, m

const N_FREESTREAM = P_FREESTREAM / (FlowRef.k_B * T_FREESTREAM)
const X_MOL_FREESTREAM = 0.9    # mole fractions
const RHO_MOL_FREESTREAM = X_MOL_FREESTREAM * N_FREESTREAM * MASS_MOL
const RHO_ATOM_FREESTREAM = (1.0 - X_MOL_FREESTREAM) * N_FREESTREAM * MASS_ATOM

# the reference density is built from the reference mass m_ref = MASS_MOL rather
# than from the mixture density, so that the scaled species masses are O(1)
const REF_Q = p_T_rho_L(P_FREESTREAM, T_FREESTREAM, N_FREESTREAM * MASS_MOL, L_REF)

# scaling for the reaction rate coefficients; equals n_ref * t_ref
const K_REF = REF_Q.m_ref * REF_Q.n_ref^2 * REF_Q.t_ref / REF_Q.rho_ref

# Arrhenius dissociation rate coefficients, k = A * T^n * exp(-E_DISS / T), m^3 / s
const CHEM_MULT = 1.0           # scales both rates, to speed up or freeze the chemistry
const A_m_a = 1.6605e-8 * CHEM_MULT  # molecule-atom dissociation
const n_m_a = -1.5
const A_m_m = 3.321e-9 * CHEM_MULT   # molecule-molecule dissociation
const n_m_m = -1.5

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

Nx = 30
Ny = Nx
polydeg = 3
cfl = 0.5

mapping = (xi, eta) -> mapping_full(xi, eta, 1.0, [1.32, 1.05, 2.25])

trees_per_dimension = (Nx, Ny)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = polydeg, initial_refinement_level = 0,
                 mapping = mapping,
                 periodicity = (false, false))

###############################################################################
# equations

# vibrational levels of O2 up to the dissociation energy, anharmonic oscillator
const E_VIBR_ARR = generate_e_vibr_arr_anharmonic_cutoff_K(Θ_VIBR, Θ_VIBR_ANH, E_DISS)

# internal degrees of freedom only; the translational parts are added by the
# equations themselves. The formation energies enter as constant offsets.
e_int_mol = T -> e_rot_cont(MASS_MOL, T) + e_vibr_from_array(MASS_MOL, E_VIBR_ARR, T) +
                 E_FORM_O2 * FlowRef.k_B / MASS_MOL
c_int_mol = T -> c_rot_cont(MASS_MOL, T) + c_vibr_from_array(MASS_MOL, E_VIBR_ARR, T)

e_int_atom = T -> zero(T) + E_FORM_O * FlowRef.k_B / MASS_ATOM
c_int_atom = T -> zero(T)

equations = CompressibleEulerEquationsMs1T2D(REF_Q,
                                             [MASS_MOL, MASS_ATOM],
                                             [e_int_mol, e_int_atom],
                                             [c_int_mol, c_int_atom];
                                             T_min = 10.0, T_max = 3.0e4, ΔT = 0.5,
                                             T_tol = 1e-11, min_T_jump = 1e-6)

###############################################################################
# initial and boundary conditions, source terms

# primitive variables are (v1, v2, T, rho_1, ..., rho_NCOMP)
@inline function initial_condition_supersonic_flow(x, t,
                                                   equations::CompressibleEulerEquationsMs1T2D)
    prim = SVector(V1_FREESTREAM / REF_Q.v_ref,
                   V2_FREESTREAM / REF_Q.v_ref,
                   T_FREESTREAM / REF_Q.T_ref,
                   RHO_MOL_FREESTREAM / REF_Q.rho_ref,
                   RHO_ATOM_FREESTREAM / REF_Q.rho_ref)
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

@inline function source_terms_Arrhenius_diss_only(u, x, t,
                                                  equations::CompressibleEulerEquationsMs1T2D)
    thermodata = equations.thermodata

    # the rate coefficients are strongly temperature dependent, so `T` has to be
    # recovered from the internal energy here, in Kelvin
    T = Trixi.temperature(u, equations) * REF_Q.T_ref

    # scaled number densities; `mass` and `inv_mass` are scaled by REF_Q.m_ref
    n_mol = u[4] * thermodata.inv_mass[1]
    n_atom = u[5] * thermodata.inv_mass[2]

    exp_E_diss = exp(-E_DISS / T)
    k_diss_m_a = A_m_a * T^n_m_a * exp_E_diss
    k_diss_m_m = A_m_m * T^n_m_m * exp_E_diss

    dn_mol = -(k_diss_m_a * n_mol * n_atom + k_diss_m_m * n_mol * n_mol) * K_REF
    dn_atom = -2 * dn_mol  # each molecule lost yields two atoms

    # the formation energies are part of e_i(T), so the total energy, thermal plus
    # chemical, is conserved: the endothermic reaction shows up as a drop in T
    drho_e = 0.0
    # the velocity is unchanged, and the mass source terms cancel by construction

    drho_mol = dn_mol * thermodata.mass[1]
    drho_atom = dn_atom * thermodata.mass[2]

    return SVector(0.0, 0.0, drho_e, drho_mol, drho_atom)
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
                                    boundary_conditions = boundary_conditions,
                                    source_terms = source_terms_Arrhenius_diss_only)

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

    return SVector(v1 * REF_Q.v_ref, v2 * REF_Q.v_ref, p * REF_Q.p_ref, T * REF_Q.T_ref,
                   u[4] * REF_Q.rho_ref, u[5] * REF_Q.rho_ref, sqrt(v1^2 + v2^2) / a)
end

function Trixi.varnames(::typeof(cons2prim_scaled),
                        ::CompressibleEulerEquationsMs1T2D)
    return ("v1", "v2", "p", "T", "rho_mol", "rho_atom", "M")
end

outpref = joinpath("out", "2species_cylinder")
outdir = joinpath(outpref, string(Nx) * "_" * string(Ny), string(polydeg), "hdf")
outdirrest = joinpath(outpref, string(Nx) * "_" * string(Ny), string(polydeg), "restart")

###############################################################################
# ODE and callbacks

tspan = (0.0, 2.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
# qualified because both Trixi and AeroTrixi export an `AnalysisCallback`; AeroTrixi's
# adds surface pointwise analysis, which is not used here
analysis_callback = Trixi.AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim_scaled,
                                     output_directory = outdir)

save_restart = SaveRestartCallback(interval = 10000, output_directory = outdirrest)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 0.5,
                                          alpha_min = 0.001,
                                          alpha_smooth = true,
                                          variable = density_pressure)

amr_controller = ControllerThreeLevel(semi, amr_indicator;
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.175,
                                      max_level = 4, max_threshold = 0.35)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5000,
                           adapt_initial_condition = false,
                           adapt_initial_condition_only_refine = false)

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        save_restart,
                        amr_callback,
                        stepsize_callback)

sol = solve(ode, SSPRK932();
            dt = 1.0, # overwritten by the stepsize_callback
            adaptive = false, # SSPRK932 is adaptive by default, which the callback forbids
            maxiters = 9999999, ode_default_options()...,
            callback = callbacks);
