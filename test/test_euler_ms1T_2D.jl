module TestEulerMs1T2D

using Test
using Trixi
using Trixi: flux_chandrashekar, CompressibleEulerEquations2D,
             CompressibleEulerMulticomponentEquations2D
using AeroTrixi

using AeroTrixi: CompressibleEulerEquationsMs1T2D, ReferenceFlowQuantities, k_B,
                 e_rot_cont, c_rot_cont, e_vibr_iho, c_vibr_iho,
                 cons2prim_with_index, get_index_lower_fracpos, get_gamma,
                 energy_internal, energy_kinetic, flux_oblapenko, SVector

using Trixi: entropy, entropy_math, entropy_thermodynamic, cons2entropy

# ------------------------------------------------------------------------------
# Two-species O2 / O flow in thermal equilibrium.
#
# The molecule carries rotational and vibrational energy, the atom none, so the
# atomic energy and specific heat are exactly linear in T and reproduced to
# machine precision by the tables while the molecular ones incur interpolation
# error. All reference quantities are built from (p, T, rho) so that the free
# stream is O(1), mirroring how FlowRef.jl scales the equations.
# ------------------------------------------------------------------------------

const MASS_MOL = 5.3134e-26   # O2
const MASS_ATOM = MASS_MOL / 2 # O
const Θ_VIBR = 2273.54        # O2 characteristic vibrational temperature, K

const T_REF = 901.0
const P_REF = 1500.0

# reference set consistent with m_ref = MASS_MOL, as p_T_rho_L would produce
function make_ref_q(m_ref)
    n_ref = P_REF / (k_B * T_REF)
    rho_ref = n_ref * m_ref
    v_ref = sqrt(P_REF / rho_ref)
    e_ref = k_B * T_REF / m_ref
    c_v_ref = k_B / m_ref
    return ReferenceFlowQuantities(P_REF, T_REF, rho_ref, v_ref, n_ref, m_ref,
                                   e_ref, c_v_ref, c_v_ref, 1.0, 1.0, 1.0, 1.0, 1.0)
end

const REF_Q = make_ref_q(MASS_MOL)

e_int_mol(T) = e_rot_cont(MASS_MOL, T) + e_vibr_iho(MASS_MOL, Θ_VIBR, T)
c_int_mol(T) = c_rot_cont(MASS_MOL, T) + c_vibr_iho(MASS_MOL, Θ_VIBR, T)
e_int_atom(T) = 0.0 * T   # atoms have no internal energy
c_int_atom(T) = 0.0 * T

# full specific energy / specific heat of a species, translational part included
e_full_mol(T) = e_int_mol(T) + 1.5 * k_B * T / MASS_MOL
e_full_atom(T) = e_int_atom(T) + 1.5 * k_B * T / MASS_ATOM
c_full_mol(T) = c_int_mol(T) + 1.5 * k_B / MASS_MOL
c_full_atom(T) = c_int_atom(T) + 1.5 * k_B / MASS_ATOM

const EQUATIONS = CompressibleEulerEquationsMs1T2D(REF_Q, [MASS_MOL, MASS_ATOM],
                                                   [e_int_mol, e_int_atom],
                                                   [c_int_mol, c_int_atom];
                                                   T_tol = 1e-11, min_T_jump = 1e-5)

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------

# scaled conservative state from physical (T [K], velocities [m/s], pressure [Pa]
# and molar fraction of the molecule)
function scaled_cons(T, v1, v2, p, x_mol; ref_q = REF_Q)
    x_atom = 1.0 - x_mol
    n = p / (k_B * T)
    rho = MASS_MOL * x_mol * n + MASS_ATOM * x_atom * n
    Y_mol = MASS_MOL * x_mol * n / rho
    Y_atom = MASS_ATOM * x_atom * n / rho

    e_int = Y_mol * e_full_mol(T) + Y_atom * e_full_atom(T)
    e_total = e_int + 0.5 * (v1^2 + v2^2)

    return SVector(rho * v1 / (ref_q.rho_ref * ref_q.v_ref),
                   rho * v2 / (ref_q.rho_ref * ref_q.v_ref),
                   rho * e_total / (ref_q.rho_ref * ref_q.e_ref),
                   rho * Y_mol / ref_q.rho_ref,
                   rho * Y_atom / ref_q.rho_ref)
end

@testset "CompressibleEulerEquationsMs1T2D" begin
    @testset "primitive quantities of a scaled state" begin
        for T in (2000.0, 5000.5, 11301.1), x_mol in (1e-7, 0.3, 0.5, 0.7, 1.0 - 1e-7)
            v1, v2, p = -600.0, 780.0, 22222.0
            u = scaled_cons(T, v1, v2, p, x_mol)

            n = p / (k_B * T)
            rho = MASS_MOL * x_mol * n + MASS_ATOM * (1 - x_mol) * n
            e_int = (MASS_MOL * x_mol * n * e_full_mol(T) +
                     MASS_ATOM * (1 - x_mol) * n * e_full_atom(T)) / rho

            @test density(u, EQUATIONS)≈rho / REF_Q.rho_ref rtol=1e-12
            @test energy_kinetic(u, EQUATIONS)≈0.5 * rho * (v1^2 + v2^2) /
                                               (REF_Q.e_ref * REF_Q.rho_ref) rtol=1e-12
            @test energy_internal(u, EQUATIONS)≈e_int * rho /
                                                (REF_Q.e_ref * REF_Q.rho_ref) rtol=1e-11
            @test energy_internal(u, EQUATIONS) + energy_kinetic(u, EQUATIONS)≈u[3] rtol=1e-12

            @test temperature(u, EQUATIONS)≈T / REF_Q.T_ref rtol=2e-9

            # pressure from u and from a known T must agree, and both give p
            p_from_u = pressure(u, EQUATIONS)
            p_from_T = pressure(T / REF_Q.T_ref, u, EQUATIONS)
            @test p_from_u≈p / REF_Q.p_ref rtol=5e-9
            @test p_from_T≈p / REF_Q.p_ref rtol=1e-12
        end
    end

    @testset "cons2prim / prim2cons round-trip" begin
        for T in (2000.0, 5000.5, 11301.1), x_mol in (1e-7, 0.3, 0.5, 0.7, 1.0 - 1e-7)
            u = scaled_cons(T, -600.0, 780.0, 22222.0, x_mol)

            prim = cons2prim(u, EQUATIONS)
            @test prim2cons(prim, EQUATIONS)≈u rtol=1e-9

            # primitive layout is (v1, v2, T, rho_1, ..., rho_NCOMP)
            @test prim[1]≈u[1] / density(u, EQUATIONS) rtol=1e-12
            @test prim[3]≈T / REF_Q.T_ref rtol=2e-9

            @test cons2prim(prim2cons(prim, EQUATIONS), EQUATIONS)≈prim rtol=1e-9
        end
    end

    @testset "cons2prim_with_index returns a consistent stencil" begin
        for T in (2000.0, 5000.5, 11301.1)
            u = scaled_cons(T, -600.0, 780.0, 22222.0, 0.4)

            ie, fe, ic, fc, prim = cons2prim_with_index(u, EQUATIONS)
            td = EQUATIONS.thermodata

            # the returned temperature must sit at the returned energy stencil
            T_stencil = td.T_arr[ie] * (1 - fe) + fe * td.T_arr[ie + 1]
            @test T_stencil≈prim[3] rtol=1e-13

            # and the indices must match a fresh lookup at that temperature
            @test (ie, fe, ic, fc) == get_index_lower_fracpos(prim[3], td)
        end
    end

    @testset "adiabatic index of a pure atomic gas" begin
        # a monatomic gas without internal energy has gamma = 5/3 exactly
        T = 6000.0
        u = scaled_cons(T, 0.0, 0.0, 22222.0, 0.0)  # pure atoms
        _, _, ic, fc, _ = cons2prim_with_index(u, EQUATIONS)
        rho_inv = 1.0 / density(u, EQUATIONS)
        @test get_gamma(u, rho_inv, ic, fc, EQUATIONS.thermodata)≈5 / 3 rtol=1e-6
    end

    # --------------------------------------------------------------------------
    # `flux_oblapenko` is the entropy-conservative two-point flux. `FluxRotated`
    # evaluates it in a frame rotated so the normal points along x, using the
    # `rotate_to_x` / `rotate_from_x` we provide; the direct normal-direction
    # method must reproduce that for every direction.
    # --------------------------------------------------------------------------

    directions() = ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
                    [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5],
                    [0.2, 0.8], [-0.2, 0.8], [0.2, -0.8], [-0.2, -0.8],
                    [0.35, 0.7], [-0.35, 0.7], [0.35, -0.7], [-0.35, -0.7])

    @testset "flux_oblapenko is rotationally invariant" begin
        flux_rotated = FluxRotated(flux_oblapenko)

        u_ll = scaled_cons(4003.5, -600.0, 780.0, 22222.0, 0.35)
        u_rr = scaled_cons(5200.5, -750.0, 858.0, 18000.0, 0.55)

        for dir in directions()
            n = SVector{2}(dir ./ sqrt(dir[1]^2 + dir[2]^2))
            @test flux_oblapenko(u_ll, u_rr, n, EQUATIONS)≈flux_rotated(u_ll, u_rr, n,
                                                                        EQUATIONS) atol=1e-13
        end
    end

    @testset "rotate_from_x inverts rotate_to_x" begin
        u = scaled_cons(4003.5, -600.0, 780.0, 22222.0, 0.35)
        for dir in directions()
            n = SVector{2}(dir ./ sqrt(dir[1]^2 + dir[2]^2))
            @test AeroTrixi.rotate_from_x(AeroTrixi.rotate_to_x(u, n, EQUATIONS), n,
                                          EQUATIONS)≈u atol=1e-14
        end
    end

    @testset "orientation and axis-aligned normal fluxes agree" begin
        u_ll = scaled_cons(4003.5, -600.0, 780.0, 22222.0, 0.35)
        u_rr = scaled_cons(5200.5, -750.0, 858.0, 18000.0, 0.55)

        # flux_oblapenko(orientation=1) must equal flux_oblapenko(normal=[1,0]) etc.
        @test flux_oblapenko(u_ll, u_rr, 1, EQUATIONS)≈flux_oblapenko(u_ll, u_rr,
                                                                      SVector(1.0, 0.0),
                                                                      EQUATIONS) atol=1e-13
        @test flux_oblapenko(u_ll, u_rr, 2, EQUATIONS)≈flux_oblapenko(u_ll, u_rr,
                                                                      SVector(0.0, 1.0),
                                                                      EQUATIONS) atol=1e-13

        # consistency: equal states give the physical flux
        u = scaled_cons(4003.5, -600.0, 780.0, 22222.0, 0.35)
        @test flux_oblapenko(u, u, 1, EQUATIONS)≈flux(u, 1, EQUATIONS) atol=1e-10
        @test flux_oblapenko(u, u, 2, EQUATIONS)≈flux(u, 2, EQUATIONS) atol=1e-10
    end

    # --------------------------------------------------------------------------
    # Entropy variables and entropy conservation. `cons2entropy` is the gradient of
    # `entropy`, and `flux_oblapenko` conserves that entropy.
    # --------------------------------------------------------------------------
    @testset "entropy variables and conservation" begin
        states = (scaled_cons(2000.0, 0.3, -0.2, 20000.0, 0.35),
                  scaled_cons(6000.0, 0.1, 0.4, 15000.0, 0.55),
                  scaled_cons(9000.5, -0.25, 0.15, 30000.0, 0.5))

        @testset "wrapper consistency" begin
            for u in states
                rho = density(u, EQUATIONS)
                @test entropy(u, EQUATIONS) == entropy_math(u, EQUATIONS)
                @test entropy_math(u, EQUATIONS)≈-rho *
                                                 entropy_thermodynamic(u, EQUATIONS) rtol=1e-13
                @test entropy_thermodynamic(u, EQUATIONS)≈entropy_thermodynamic(u, rho,
                                                                                EQUATIONS) rtol=1e-13
            end
        end

        @testset "momentum and energy entropy variables" begin
            # the first three entropy variables are exactly (v1/T, v2/T, -1/T)
            for u in states
                w = cons2entropy(u, EQUATIONS)
                _, _, _, _, prim = cons2prim_with_index(u, EQUATIONS)
                v1, v2, T = prim[1], prim[2], prim[3]
                @test w[1]≈v1 / T rtol=1e-11
                @test w[2]≈v2 / T rtol=1e-11
                @test w[3]≈-1 / T rtol=1e-11
            end
        end

        @testset "cons2entropy is the gradient of the entropy" begin
            basis(i) = SVector{5}(ntuple(k -> k == i ? 1.0 : 0.0, 5))
            for u in states
                grad = SVector{5}(ntuple(5) do i
                    h = 1e-6 * max(abs(u[i]), 1.0)
                    (entropy(u + h * basis(i), EQUATIONS) -
                     entropy(u - h * basis(i), EQUATIONS)) / (2h)
                end)
                @test cons2entropy(u, EQUATIONS)≈grad rtol=1e-5
            end
        end

        @testset "flux_oblapenko is entropy conservative" begin
            # Tadmor condition: (w_ll - w_rr) . f* = psi_ll - psi_rr, with the
            # entropy-flux potential psi_j = w . f_phys_j - v_j * entropy
            function psi(u, orientation)
                rho = density(u, EQUATIONS)
                v = orientation == 1 ? u[1] / rho : u[2] / rho
                return sum(cons2entropy(u, EQUATIONS) .* flux(u, orientation, EQUATIONS)) -
                       v * entropy(u, EQUATIONS)
            end

            pairs = ((states[1], states[2]),  # large temperature jump
                     (states[2], states[3]),
                     (states[1], states[1] .* 1.0000003))  # tiny jump, midpoint branch
            for (u_ll, u_rr) in pairs, orientation in (1, 2)
                fstar = flux_oblapenko(u_ll, u_rr, orientation, EQUATIONS)
                lhs = sum((cons2entropy(u_ll, EQUATIONS) .- cons2entropy(u_rr, EQUATIONS)) .*
                          fstar)
                rhs = psi(u_ll, orientation) - psi(u_rr, orientation)
                @test lhs≈rhs atol=1e-9
            end
        end
    end

    # --------------------------------------------------------------------------
    # For a calorically perfect gas (constant c_v) `flux_oblapenko` reduces to the
    # entropy-conservative flux of a fixed-gamma gas, so it must reproduce Trixi's
    # `flux_chandrashekar` for the equivalent ideal-gas equations.
    #
    # The reference quantities below give k_B = 1 in scaled units (m_ref = k_B),
    # so a species of scaled mass m has R = 1/m and, with a constant internal
    # specific heat c_int, a total c_v = 3/2 / m + c_int and gamma = 1 + R / c_v.
    # --------------------------------------------------------------------------
    @testset "constant c_v matches flux_chandrashekar" begin
        # scaled masses 1 and 2, i.e. physical masses k_B and 2 k_B
        m1, m2 = k_B, 2 * k_B
        ref_q_unit = ReferenceFlowQuantities(1.0, 1.0, 1.0, 1.0, 1.0 / k_B, k_B,
                                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        # internal contribution (constant c_v), scaled value = physical value here
        # since c_v_ref = k_B / m_ref = 1:
        #   species 1: c_int = 1   -> c_v = 3/2 + 1   = 2.5, R = 1,   gamma = 1.4
        #   species 2: c_int = 0   -> c_v = 3/4       = 0.75, R = 0.5, gamma = 5/3
        e_int_1(T) = 1.0 * T
        c_int_1(T) = 1.0
        e_int_2(T) = 0.0 * T
        c_int_2(T) = 0.0

        cv1, cv2 = 2.5, 0.75
        gamma1, gamma2 = 1.4, 5 / 3
        R1, R2 = 1.0, 0.5

        # small min_T_jump so the divided-difference branch (T_ll != T_rr) is taken
        eq_ms_1sp = CompressibleEulerEquationsMs1T2D(ref_q_unit, [m1], [e_int_1], [c_int_1];
                                                     min_T_jump = 1e-9)
        eq_ms_mc = CompressibleEulerEquationsMs1T2D(ref_q_unit, [m1, m2],
                                                    [e_int_1, e_int_2], [c_int_1, c_int_2];
                                                    min_T_jump = 1e-9)

        eq_ref_1sp = CompressibleEulerEquations2D(gamma1)
        eq_ref_mc = CompressibleEulerMulticomponentEquations2D(gammas = (gamma1, gamma2),
                                                               gas_constants = (R1, R2))

        # conservative state in the shared (rho_v1, rho_v2, rho_e, rho_1, ...) layout
        function ms_state(T, v1, v2, rhos, cvs)
            rho = sum(rhos)
            rho_e = sum(rhos .* cvs) * T + 0.5 * rho * (v1^2 + v2^2)
            return SVector(rho * v1, rho * v2, rho_e, rhos...)
        end

        @testset "single species vs CompressibleEulerEquations2D" begin
            u_ll = ms_state(2000.0, 0.3, -0.2, (0.7,), (cv1,))
            u_rr = ms_state(6000.0, 0.1, 0.4, (0.9,), (cv1,))

            # Trixi single-species order is (rho, rho_v1, rho_v2, rho_e)
            to_ref(u) = SVector(u[4], u[1], u[2], u[3])
            from_ms(f) = SVector(f[4], f[1], f[2], f[3])
            r_ll, r_rr = to_ref(u_ll), to_ref(u_rr)

            for orientation in (1, 2)
                f_ms = flux_oblapenko(u_ll, u_rr, orientation, eq_ms_1sp)
                f_ref = flux_chandrashekar(r_ll, r_rr, orientation, eq_ref_1sp)
                @test from_ms(f_ms)≈f_ref rtol=1e-11
            end

            for dir in ([1.0, 0.0], [0.35, 0.7], [-0.6, 0.8])
                n = SVector{2}(dir ./ sqrt(dir[1]^2 + dir[2]^2))
                f_ms = flux_oblapenko(u_ll, u_rr, n, eq_ms_1sp)
                f_ref = flux_chandrashekar(r_ll, r_rr, n, eq_ref_1sp)
                @test from_ms(f_ms)≈f_ref rtol=1e-11
            end
        end

        @testset "multi species vs CompressibleEulerMulticomponentEquations2D" begin
            # same conservative-variable layout, no reordering needed
            u_ll = ms_state(2000.0, 0.3, -0.2, (0.7, 0.5), (cv1, cv2))
            u_rr = ms_state(6000.0, 0.1, 0.4, (0.9, 0.3), (cv1, cv2))

            for orientation in (1, 2)
                f_ms = flux_oblapenko(u_ll, u_rr, orientation, eq_ms_mc)
                f_ref = flux_chandrashekar(u_ll, u_rr, orientation, eq_ref_mc)
                @test f_ms≈f_ref rtol=1e-11
            end
        end
    end
end

end # module
