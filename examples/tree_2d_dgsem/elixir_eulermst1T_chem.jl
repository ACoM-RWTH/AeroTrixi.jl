using OrdinaryDiffEqSSPRK
using Trixi
using AeroTrixi
using FlowRef

###############################################################################
# Dissociating two-species O2/O mixture in thermal equilibrium: a uniform state
# in a periodic box, relaxed by O2 + M -> 2O + M with temperature-independent
# rate coefficients. The flow field stays uniform, so the solution is that of the
# reaction ODE alone, which is available in closed form: `initial_condition` is
# the analytical solution at time `t`, and the error norms reported by the
# analysis callback are therefore the error of the chemistry integration.

# free-stream conditions
const MASS_MOL = 5.3134e-26     # O2 molecular mass, kg
const MASS_ATOM = MASS_MOL / 2  # O atomic mass, kg

const Θ_VIBR = 2273.54          # O2 characteristic vibrational temperature, K
const E_DISS = 59364.8          # O2 dissociation energy, K
const E_FORM_O2 = 0.0           # formation energies, K
const E_FORM_O = 0.5 * E_DISS

const V1_FREESTREAM = 0.0       # m / s
const V2_FREESTREAM = 0.0
const T_FREESTREAM = 12000.0    # K
const N_FREESTREAM = 1.0e23     # total number density, 1 / m^3
const P_FREESTREAM = N_FREESTREAM * FlowRef.k_B * T_FREESTREAM

const X_MOL = 0.5               # mole fractions
const X_ATOM = 1.0 - X_MOL
const RHO_MOL = X_MOL * N_FREESTREAM * MASS_MOL
const RHO_ATOM = X_ATOM * N_FREESTREAM * MASS_ATOM

const L_REF = 1.0               # m

# the reference density is built from the reference mass m_ref = MASS_MOL rather
# than from the mixture density, so that the scaled species masses are O(1)
const REF_Q = p_T_rho_L(P_FREESTREAM, T_FREESTREAM, N_FREESTREAM * MASS_MOL, L_REF)

# scaling for the reaction rate coefficients; equals n_ref * t_ref
const K_REF = REF_Q.m_ref * REF_Q.n_ref^2 * REF_Q.t_ref / REF_Q.rho_ref

# Dissociation rate coefficients, m^3 / s. These are constants rather than the
# strongly temperature-dependent physical values, which is what makes the species
# equations solvable in closed form. They are also several orders of magnitude
# below the physical O2 rates at 12000 K: with the latter the mixture dissociates
# in ~2e-8 s, far below the acoustic time L_REF / v_ref ~ 6e-4 s of the box, and
# nothing of the reaction would be resolved on the CFL time step. As chosen here,
# 30% of the molecules dissociate over the run.
const A_m_a = 9.0e-21           # molecule-atom dissociation
const A_m_m = 3.0e-21           # molecule-molecule dissociation

###############################################################################
# equations

# vibrational levels of O2 up to the dissociation energy, harmonic oscillator
const E_VIBR_ARR = generate_e_vibr_arr_harmonic_cutoff_K(Θ_VIBR, E_DISS)

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
                                             T_min = 10.0, T_max = 4.0e4, ΔT = 1.0,
                                             T_tol = 1e-11, min_T_jump = 1e-6)

###############################################################################
# initial condition and source terms

# the state at t = 0; primitive variables are (v1, v2, T, rho_1, ..., rho_NCOMP)
const U_INITIAL = prim2cons(SVector(V1_FREESTREAM / REF_Q.v_ref,
                                    V2_FREESTREAM / REF_Q.v_ref,
                                    T_FREESTREAM / REF_Q.T_ref,
                                    RHO_MOL / REF_Q.rho_ref,
                                    RHO_ATOM / REF_Q.rho_ref),
                            equations)

# scaled number densities at t = 0 and the conserved number of oxygen nuclei
const N_MOL_0 = X_MOL * N_FREESTREAM / REF_Q.n_ref
const N_ATOM_0 = X_ATOM * N_FREESTREAM / REF_Q.n_ref
const C_NUCLEI = 2 * N_MOL_0 + N_ATOM_0

# with `n_atom = C_NUCLEI - 2 * n_mol` the reaction ODE closes on `n_mol` alone,
#   dn/dt = -n * (α - β * n),   α = K_REF * A_m_a * C_NUCLEI,
#                               β = K_REF * (2 * A_m_a - A_m_m)
# a Riccati equation that the substitution y = 1/n turns into a linear one
const ALPHA_CHEM = K_REF * A_m_a * C_NUCLEI
const BETA_CHEM = K_REF * (2 * A_m_a - A_m_m)

@inline function n_mol_exact(t)
    return 1 /
           ((1 / N_MOL_0 - BETA_CHEM / ALPHA_CHEM) * exp(ALPHA_CHEM * t) +
            BETA_CHEM / ALPHA_CHEM)
end

# The analytical solution: the state is uniform in space, so the flux divergence
# vanishes and the momenta stay at their initial values; the reaction conserves
# the total energy, so `rho_e` does too. Only the species densities evolve, which
# is why no inversion of e(T) for the temperature at time `t` is needed here.
@inline function initial_condition_dissociation(x, t,
                                                equations::CompressibleEulerEquationsMs1T2D)
    n_mol = n_mol_exact(t)
    n_atom = C_NUCLEI - 2 * n_mol

    return SVector(U_INITIAL[1], U_INITIAL[2], U_INITIAL[3],
                   n_mol * equations.thermodata.mass[1],
                   n_atom * equations.thermodata.mass[2])
end

initial_condition = initial_condition_dissociation

@inline function source_terms_dissociation(u, x, t,
                                           equations::CompressibleEulerEquationsMs1T2D)
    thermodata = equations.thermodata

    # scaled number densities; `mass` and `inv_mass` are scaled by REF_Q.m_ref
    n_mol = u[4] * thermodata.inv_mass[1]
    n_atom = u[5] * thermodata.inv_mass[2]

    dn_mol = -(A_m_a * n_mol * n_atom + A_m_m * n_mol * n_mol) * K_REF
    dn_atom = -2 * dn_mol  # each molecule lost yields two atoms

    # the formation energies are part of e_i(T), so the total energy, thermal plus
    # chemical, is conserved: the endothermic reaction shows up as a drop in T
    drho_e = 0.0
    # the velocity is unchanged, and the mass source terms cancel by construction

    drho_mol = dn_mol * thermodata.mass[1]
    drho_atom = dn_atom * thermodata.mass[2]

    return SVector(0.0, 0.0, drho_e, drho_mol, drho_atom)
end

###############################################################################
# solver and mesh

polydeg = 3

volume_flux = flux_oblapenko
surface_flux = flux_oblapenko

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_periodic,
                                    source_terms = source_terms_dissociation)

###############################################################################
# output in dimensional units

function cons2prim_scaled(u, equations::CompressibleEulerEquationsMs1T2D)
    _, _, _, _, prim = AeroTrixi.cons2prim_with_index(u, equations)
    v1, v2, T = prim[1], prim[2], prim[3]

    p = pressure(T, u, equations)

    return SVector(v1 * REF_Q.v_ref, v2 * REF_Q.v_ref, p * REF_Q.p_ref, T * REF_Q.T_ref,
                   u[4] * REF_Q.rho_ref, u[5] * REF_Q.rho_ref)
end

function Trixi.varnames(::typeof(cons2prim_scaled),
                        ::CompressibleEulerEquationsMs1T2D)
    return ("v1", "v2", "p", "T", "rho_mol", "rho_atom")
end

###############################################################################
# ODE and callbacks

const T_MAX = 5000 * 1e-7       # s
tspan = (0.0, T_MAX / REF_Q.t_ref)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

alive_callback = AliveCallback(alive_interval = 10)
# qualified because both Trixi and AeroTrixi export an `AnalysisCallback`; AeroTrixi's
# adds surface pointwise analysis, which is not used here. Note that the reaction
# produces entropy, so `∑∂S/∂U ⋅ Uₜ` is not expected to vanish here
analysis_callback = Trixi.AnalysisCallback(semi, interval = 10,
                                           extra_analysis_integrals = (entropy,
                                                                       Trixi.entropy_timederivative))

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim_scaled)

# the time step is prescribed by the CFL condition rather than by an error
# estimator: an adaptive step sequence depends on the time integrator version and
# makes the solution irreproducible below the integrator tolerance
stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

sol = solve(ode, SSPRK43();
            dt = 1.0, # overwritten by the stepsize_callback
            adaptive = false, # SSPRK43 is adaptive by default, which the callback forbids
            ode_default_options()..., callback = callbacks);
