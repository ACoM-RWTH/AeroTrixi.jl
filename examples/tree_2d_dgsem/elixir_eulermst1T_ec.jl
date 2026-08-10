using OrdinaryDiffEqSSPRK
using Trixi
using AeroTrixi
using FlowRef

###############################################################################
# Entropy-conservation test for a two-species O2/O mixture in thermal
# equilibrium: a weak blast wave in a doubly periodic box. `flux_oblapenko` is
# used both as the volume flux of the flux-differencing volume integral and as
# the surface flux, so the semidiscretisation is entropy conservative and
# `∑∂S/∂U ⋅ Uₜ` reported by the analysis callback should stay at round-off.

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
# initial condition

# The perturbation is the one of `Trixi.initial_condition_weak_blast_wave`: inside
# the circle r = 0.5 the density is raised by a factor 1.1691 and a radial velocity
# of 0.1882 (in units of `v_ref`) is added. The pressure ratio 1.245 / 1.1691 of the
# original becomes a temperature ratio here, the composition being left untouched,
# so that p = n T scales the same way.
#
# primitive variables are (v1, v2, T, rho_1, ..., rho_NCOMP)
@inline function initial_condition_weak_blast_wave(x, t,
                                                   equations::CompressibleEulerEquationsMs1T2D)
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    inside = r <= 0.5
    rho_factor = inside ? 1.1691 : 1.0
    T_factor = inside ? 1.245 / 1.1691 : 1.0
    v_perturbation = inside ? 0.1882 : 0.0

    prim = SVector(V1_FREESTREAM / REF_Q.v_ref + v_perturbation * cos_phi,
                   V2_FREESTREAM / REF_Q.v_ref + v_perturbation * sin_phi,
                   T_factor * T_FREESTREAM / REF_Q.T_ref,
                   rho_factor * RHO_MOL / REF_Q.rho_ref,
                   rho_factor * RHO_ATOM / REF_Q.rho_ref)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_weak_blast_wave

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
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_periodic)

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
# adds surface pointwise analysis, which is not used here
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
