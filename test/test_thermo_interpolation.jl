module TestThermoInterpolation

using Test
using AeroTrixi

using AeroTrixi: ThermoData1T, ncomponents, eachcomponent,
                 get_index_lower_fracpos,
                 energy_component, c_v_component, energy, c_v,
                 energy_from_rho_vec,
                 entropy_c_v_integral_component, entropy_c_v_integral,
                 limit_T_low_rho_inv, temperature_rho_inv,
                 temperature_rho_inv_with_index, get_gamma,
                 ReferenceFlowQuantities, SVector, k_B

# ------------------------------------------------------------------------------
# Test model
#
# The internal (non-translational) contribution is given a constant specific heat,
# so that the total specific heat of species i is constant,
#     c_v_i = 3/2 * k_B / m_i + c_int_i,
# and the specific energy is exactly linear,
#     e_i(T) = c_v_i * T.
# Linear interpolation is then exact on both the energy and the c_v grid, which
# lets every quantity be compared against a closed-form expression.
# ------------------------------------------------------------------------------

const MASSES = [4.65e-26, 5.31e-26]  # ~N2, ~O2
const C_INT = [2.0 * k_B / MASSES[1], 1.5 * k_B / MASSES[2]]
const C_V_TOT = [1.5 * k_B / MASSES[1] + C_INT[1], 1.5 * k_B / MASSES[2] + C_INT[2]]

const E_C_FUNS = [(m, T, _) -> (C_INT[1] * T, C_INT[1]),
                  (m, T, _) -> (C_INT[2] * T, C_INT[2])]

const T_MIN = 100.0
const T_MAX = 5000.0
const ΔT = 10.0
const N_T = trunc(Int, (T_MAX - T_MIN) / ΔT) + 1

# mass fractions used to build the conservative state
const Y = [0.4, 0.6]

# reference quantities equal to one, i.e. scaling is the identity
identity_ref_q() = ReferenceFlowQuantities(ntuple(_ -> 1.0, 14)...)

# reference quantities that scale T, e, c_v and m by distinct, non-unit factors
function scaled_ref_q()
    T_ref = 273.15
    m_ref = 5.0e-26
    return ReferenceFlowQuantities(1.0, T_ref, 1.0, 1.0, 1.0, m_ref,
                                   k_B * T_ref / m_ref, k_B / m_ref, k_B / m_ref,
                                   1.0, 1.0, 1.0, 1.0, 1.0)
end

function build(ref_q, offset)
    return ThermoData1T(ref_q, copy(MASSES), E_C_FUNS;
                        T_min = T_MIN, T_max = T_MAX, ΔT = ΔT,
                        cv_table_offset = offset)
end

# conservative state (rho_v1, rho_v2, rho_e, rho_1, ..., rho_NCOMP); rho_e is unused
# by everything under test here, so it is left at zero
state(rho) = SVector(0.0, 0.0, 0.0, rho * Y[1], rho * Y[2])

# reference implementation of the index lookup, using an independent division and
# floor per grid instead of the single-comparison shortcut in `get_index_lower_fracpos`
function reference_indices(T, T_min, ΔT, offset)
    pos_e = (T - T_min) / ΔT
    index_e = floor(Int, pos_e)
    frac_e = pos_e - index_e

    T_min_c = offset ? T_min - 0.5 * ΔT : T_min
    pos_c = (T - T_min_c) / ΔT
    index_c = floor(Int, pos_c)
    frac_c = pos_c - index_c

    return index_e + 1, frac_e, index_c + 1, frac_c
end

# temperatures the solver is allowed to produce, see the clamping in
# `temperature_rho_inv`
T_lo() = 1.0001 * T_MIN
T_hi() = 0.9999 * T_MAX
sample_temperatures(n) = range(T_lo(), T_hi(); length = n)

@testset "ThermoData1T" begin
    @testset "table sizes and grids" begin
        for offset in (false, true)
            td = build(identity_ref_q(), offset)

            @test ncomponents(td) == length(MASSES)
            @test eachcomponent(td) == Base.OneTo(length(MASSES))

            # the energy table is unaffected by the c_v offset
            @test size(td.e_arr) == (N_T, length(MASSES))
            @test length(td.T_arr) == N_T
            @test td.T_arr[1] ≈ T_MIN
            @test td.T_arr[end] ≈ T_MAX
            @test td.T_min_E ≈ T_MIN
            @test td.T_max_E ≈ T_MAX

            n_c = offset ? N_T + 2 : N_T
            @test size(td.c_v_arr) == (n_c, length(MASSES))
            @test size(td.int_c_v_over_t_arr) == (n_c, length(MASSES))
            @test length(td.T_c_arr) == n_c
            @test length(td.T_c_arr_inv) == n_c
            @test td.T_c_arr_inv ≈ 1.0 ./ td.T_c_arr

            # c_v grid is shifted by -ΔT/2 when the offset is used, and by nothing otherwise
            T_c_min = offset ? T_MIN - 0.5 * ΔT : T_MIN
            @test td.T_min_c_v ≈ T_c_min
            @test td.T_max_c_v ≈ T_c_min + (n_c - 1) * ΔT
            @test td.T_c_arr ≈ [T_c_min + (i - 1) * ΔT for i in 1:n_c]

            # the c_v grid has to bracket the whole energy range
            @test td.T_c_arr[1] <= T_MIN
            @test td.T_c_arr[end] >= T_MAX

            @test td.ΔT ≈ ΔT
            @test td.inv_ΔT ≈ 1.0 / ΔT
        end
    end

    @testset "e_min_arr" begin
        td = build(identity_ref_q(), true)
        @test length(td.e_min_arr) == length(MASSES)
        # e_i(T) is increasing, so the minimum is attained at T_min
        @test td.e_min_arr ≈ [C_V_TOT[i] * T_MIN for i in eachindex(MASSES)]
    end

    @testset "get_index_lower_fracpos" begin
        for offset in (false, true)
            td = build(identity_ref_q(), offset)
            n_c = size(td.c_v_arr, 1)

            for T in sample_temperatures(50)
                index_e, frac_e, index_c, frac_c = get_index_lower_fracpos(T, td)
                ref_e, ref_fe, ref_c, ref_fc = reference_indices(T, T_MIN, ΔT, offset)

                # must agree with the independent two-division reference
                @test index_e == ref_e
                @test index_c == ref_c
                @test frac_e≈ref_fe atol=1e-12
                @test frac_c≈ref_fc atol=1e-12

                # both interpolation stencils must stay inside their tables
                @test 1 <= index_e && index_e + 1 <= N_T
                @test 1 <= index_c && index_c + 1 <= n_c

                @test 0.0 <= frac_e < 1.0
                @test 0.0 <= frac_c < 1.0
            end
        end
    end

    @testset "get_index_lower_fracpos: energy and c_v coincide without offset" begin
        td = build(identity_ref_q(), false)
        for T in sample_temperatures(10)
            index_e, frac_e, index_c, frac_c = get_index_lower_fracpos(T, td)
            @test index_e == index_c
            @test frac_e == frac_c
        end
    end

    @testset "get_index_lower_fracpos: offset is exactly half a cell" begin
        td = build(identity_ref_q(), true)
        for T in sample_temperatures(10)
            _, frac_e, index_c, frac_c = get_index_lower_fracpos(T, td)
            # the c_v position is the energy position shifted by +1/2, so the index
            # advances by one exactly when the shifted fractional position wraps
            @test index_c == (frac_e >= 0.5 ? 1 : 0) + first(get_index_lower_fracpos(T, td))
            @test frac_c≈mod(frac_e + 0.5, 1.0) atol=1e-12
        end
    end

    @testset "get_index_lower_fracpos: table edges" begin
        # inside the range the temperature solver clamps to, every stencil is valid
        for offset in (false, true)
            td = build(identity_ref_q(), offset)
            n_c = size(td.c_v_arr, 1)
            for T in (T_MIN, T_lo(), T_hi())
                index_e, _, index_c, _ = get_index_lower_fracpos(T, td)
                @test 1 <= index_e && index_e + 1 <= N_T
                @test 1 <= index_c && index_c + 1 <= n_c
            end
        end

        # exactly at T_max the upper energy stencil runs past the end of the energy
        # table; this is why `temperature_rho_inv` clamps to 0.9999 * T_max
        td_no = build(identity_ref_q(), false)
        td_off = build(identity_ref_q(), true)

        index_e, _, index_c_no, _ = get_index_lower_fracpos(T_MAX, td_no)
        @test index_e + 1 > N_T
        @test index_c_no + 1 > size(td_no.c_v_arr, 1)

        # the two extra points of the offset c_v table keep its stencil valid there,
        # so the offset table never fails earlier than the energy table does
        _, _, index_c_off, _ = get_index_lower_fracpos(T_MAX, td_off)
        @test index_c_off + 1 <= size(td_off.c_v_arr, 1)
    end

    @testset "energy interpolation" begin
        for offset in (false, true)
            td = build(identity_ref_q(), offset)
            rho = 1.7
            u = state(rho)

            for T in sample_temperatures(40)
                index_e, frac_e, _, _ = get_index_lower_fracpos(T, td)

                for i in eachcomponent(td)
                    @test energy_component(i, index_e, frac_e, td)≈C_V_TOT[i] * T rtol=1e-12
                end

                e_exact = sum(Y[i] * C_V_TOT[i] * T for i in eachcomponent(td))
                @test energy(u, 1.0 / rho, index_e, frac_e, td)≈e_exact rtol=1e-12
                @test energy_from_rho_vec(SVector(rho * Y[1], rho * Y[2]), rho, T,
                                          td)≈e_exact rtol=1e-12
            end
        end
    end

    @testset "c_v interpolation" begin
        for offset in (false, true)
            td = build(identity_ref_q(), offset)
            rho = 1.7
            u = state(rho)
            c_v_exact = sum(Y[i] * C_V_TOT[i] for i in eachcomponent(td))

            for T in sample_temperatures(40)
                _, _, index_c, frac_c = get_index_lower_fracpos(T, td)

                for i in eachcomponent(td)
                    @test c_v_component(i, index_c, frac_c, td)≈C_V_TOT[i] rtol=1e-12
                end
                @test c_v(u, 1.0 / rho, index_c, frac_c, td)≈c_v_exact rtol=1e-12
            end
        end
    end

    @testset "entropy integral" begin
        for offset in (false, true)
            td = build(identity_ref_q(), offset)
            rho = 1.7
            u = state(rho)

            # the tabulated integral starts at the first point of the c_v grid,
            # so ∫ c_v / T dT = c_v * log(T / T_c_arr[1]) for constant c_v
            T_0 = td.T_c_arr[1]
            c_v_exact = sum(Y[i] * C_V_TOT[i] for i in eachcomponent(td))

            for T in sample_temperatures(40)
                _, _, index_c, frac_c = get_index_lower_fracpos(T, td)

                for i in eachcomponent(td)
                    @test entropy_c_v_integral_component(i, index_c, T,
                                                         td)≈C_V_TOT[i] *
                                                             log(T / T_0) rtol=1e-11
                end

                @test entropy_c_v_integral(u, T, rho, td)≈c_v_exact *
                                                          log(T / T_0) rtol=1e-11
            end
        end
    end

    @testset "temperature from energy" begin
        for offset in (false, true)
            td = build(identity_ref_q(), offset)
            rho = 1.7
            u = state(rho)

            for T in sample_temperatures(40)
                e = sum(Y[i] * C_V_TOT[i] * T for i in eachcomponent(td))

                # the Newton solve has to recover T from e for any starting guess
                # inside the tabulated range
                for T0 in (T, T_lo(), T_hi(), 0.5 * (T_MIN + T_MAX))
                    @test temperature_rho_inv(u, 1.0 / rho, T0, e, td)≈T rtol=1e-9
                end

                T_sol, index_e, frac_e, index_c, frac_c = temperature_rho_inv_with_index(u,
                                                                                        1.0 /
                                                                                        rho,
                                                                                        T, e,
                                                                                        td)
                @test T_sol≈T rtol=1e-9
                @test (index_e, frac_e, index_c, frac_c) ==
                      get_index_lower_fracpos(T_sol, td)
            end
        end
    end

    @testset "temperature clamping" begin
        for offset in (false, true)
            td = build(identity_ref_q(), offset)
            rho = 1.7
            u = state(rho)
            e_mid = sum(Y[i] * C_V_TOT[i] for i in eachcomponent(td)) *
                    0.5 * (T_MIN + T_MAX)

            # out-of-range initial guesses are clamped, and a scalar is returned
            T_below = temperature_rho_inv(u, 1.0 / rho, 0.5 * T_MIN, e_mid, td)
            T_above = temperature_rho_inv(u, 1.0 / rho, 2.0 * T_MAX, e_mid, td)
            @test T_below isa Float64
            @test T_above isa Float64
            @test T_below ≈ 1.0001 * T_MIN
            @test T_above ≈ 0.9999 * T_MAX

            # the indexed variant clamps to the same temperatures and returns
            # indices consistent with them
            T_c, index_e, frac_e, index_c, frac_c = temperature_rho_inv_with_index(u,
                                                                                  1.0 / rho,
                                                                                  0.5 * T_MIN,
                                                                                  e_mid, td)
            @test T_c ≈ 1.0001 * T_MIN
            @test (index_e, frac_e, index_c, frac_c) == get_index_lower_fracpos(T_c, td)
        end
    end

    @testset "low energy limiter" begin
        for offset in (false, true)
            td = build(identity_ref_q(), offset)
            rho = 1.7
            u = state(rho)
            e_min = sum(Y[i] * C_V_TOT[i] * T_MIN for i in eachcomponent(td))

            @test limit_T_low_rho_inv(u, 1.0 / rho, 0.5 * e_min, td)
            @test limit_T_low_rho_inv(u, 1.0 / rho, e_min, td)
            @test !limit_T_low_rho_inv(u, 1.0 / rho, 1.5 * e_min, td)

            # an energy below the table minimum is clamped to the lower bound
            @test temperature_rho_inv(u, 1.0 / rho, 0.5 * (T_MIN + T_MAX), 0.5 * e_min,
                                     td) ≈ 1.0001 * T_MIN
        end
    end

    @testset "adiabatic index" begin
        for offset in (false, true)
            td = build(identity_ref_q(), offset)
            rho = 1.7
            u = state(rho)

            c_v_exact = sum(Y[i] * C_V_TOT[i] for i in eachcomponent(td))
            # R = sum_i Y_i * k_B / m_i, with k_B = 1 in these units
            R_exact = sum(Y[i] / MASSES[i] for i in eachcomponent(td))

            for T in sample_temperatures(40)
                _, _, index_c, frac_c = get_index_lower_fracpos(T, td)
                @test get_gamma(u, 1.0 / rho, index_c, frac_c,
                                td)≈(c_v_exact + R_exact) / c_v_exact rtol=1e-12
            end
        end
    end

    @testset "reference scaling" begin
        ref_q = scaled_ref_q()

        for offset in (false, true)
            td = build(ref_q, offset)
            n_c = offset ? N_T + 2 : N_T

            # grids are stored in units of T_ref
            @test td.T_min_E ≈ T_MIN / ref_q.T_ref
            @test td.T_max_E ≈ T_MAX / ref_q.T_ref
            @test td.ΔT ≈ ΔT / ref_q.T_ref
            @test td.T_arr ≈ [T_MIN + (i - 1) * ΔT for i in 1:N_T] ./ ref_q.T_ref

            T_c_min = offset ? T_MIN - 0.5 * ΔT : T_MIN
            @test td.T_c_arr ≈ [T_c_min + (i - 1) * ΔT for i in 1:n_c] ./ ref_q.T_ref

            # masses are stored in units of m_ref
            @test td.mass ≈ MASSES ./ ref_q.m_ref
            @test td.inv_mass ≈ 1.0 ./ (MASSES ./ ref_q.m_ref)

            rho = 1.7
            u = state(rho)
            c_v_exact = sum(Y[i] * C_V_TOT[i] for i in eachcomponent(td)) / ref_q.c_v_ref

            for T in sample_temperatures(40)
                T_scaled = T / ref_q.T_ref
                index_e, frac_e, index_c, frac_c = get_index_lower_fracpos(T_scaled, td)

                e_exact = sum(Y[i] * C_V_TOT[i] * T for i in eachcomponent(td)) /
                          ref_q.e_ref
                @test energy(u, 1.0 / rho, index_e, frac_e, td)≈e_exact rtol=1e-12
                @test c_v(u, 1.0 / rho, index_c, frac_c, td)≈c_v_exact rtol=1e-12

                # inverting the scaled energy has to give back the scaled temperature
                @test temperature_rho_inv(u, 1.0 / rho, T_scaled, e_exact,
                                         td)≈T_scaled rtol=1e-9
            end
        end
    end

    @testset "constructor argument checks" begin
        ref_q = identity_ref_q()

        # number of masses and of energy/specific heat functions must agree
        @test_throws AssertionError ThermoData1T{LinearInterpolation, NoCvOffset, 3}(ref_q,
                                                                                     copy(MASSES),
                                                                                     E_C_FUNS;
                                                                                     T_min = T_MIN,
                                                                                     T_max = T_MAX,
                                                                                     ΔT = ΔT)

        # the offset c_v grid starts at T_min - ΔT/2, which must stay positive
        @test_throws AssertionError ThermoData1T(ref_q, copy(MASSES), E_C_FUNS;
                                                 T_min = 0.5 * ΔT, T_max = T_MAX, ΔT = ΔT,
                                                 cv_table_offset = true)

        @test_throws ErrorException ThermoData1T(ref_q, copy(MASSES), E_C_FUNS;
                                                 T_min = T_MIN, T_max = T_MAX, ΔT = ΔT,
                                                 interpolation = :quadratic)
    end

    @testset "no mutation of mass_arr" begin
        ref_q = scaled_ref_q()

        for offset in (false, true)
            masses_copy = copy(MASSES)

            td = ThermoData1T(ref_q, masses_copy, E_C_FUNS;
                              T_min = T_MIN, T_max = T_MAX, ΔT = ΔT,
                              cv_table_offset = offset)

            # the caller's array must come back untouched, while the stored masses
            # are scaled by m_ref
            @test maximum(abs.(MASSES .- masses_copy)) < 2 * eps()
            @test maximum(abs.(masses_copy .- td.mass)) > 1.0
        end
    end
end

end # module
