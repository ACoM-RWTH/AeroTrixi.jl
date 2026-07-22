module TestSpectra

using Test
using AeroTrixi

using AeroTrixi: k_B, e_c_none, Z_vibr, avg_over_vibr_array,
                 ThermoData1T, ReferenceFlowQuantities,
                 get_index_lower_fracpos, energy_component, c_v_component

# characteristic vibrational temperature, anharmonicity and dissociation energy,
# all in K; loosely modelled on N2
const Θ = 1200.0
const Θ_A = 1.0
const E_DISS = 76350.0
const M = 1e-26

const T_LO = 2000.0
const T_MID = 8000.0
const T_HI = 30000.0

const TEMPERATURES = (T_LO, T_MID, T_HI)

# central difference of f at T, used to check specific heats against dE/dT
function central_difference(f, T; rel_step = 1e-4)
    h = rel_step * T
    return (f(T + h) - f(T - h)) / (2 * h)
end

@testset "spectra" begin
    @testset "harmonic ladder" begin
        ve_harmonic = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS)

        # the ladder is cut off at the last level below the dissociation energy
        @test ve_harmonic[end] < E_DISS
        @test ve_harmonic[end] + Θ > E_DISS

        # ground level energy is zero by default, levels are equidistant
        @test ve_harmonic[1] == 0.0
        @test ve_harmonic ≈ [(i - 1) * Θ for i in eachindex(ve_harmonic)]
        @test all(diff(ve_harmonic) .≈ Θ)
    end

    @testset "harmonic ladder: ground level offset" begin
        ve_zero = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS;
                                                        ground_level_energy_zero = true)
        ve_offset = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS;
                                                          ground_level_energy_zero = false)

        # `ground_level_energy_zero = true` is the default
        @test ve_zero == generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS)

        # with the offset the ladder carries the zero-point energy, E_i = (i + 1/2) Θ
        @test ve_offset[1] ≈ 0.5 * Θ
        @test ve_offset ≈ [(i - 0.5) * Θ for i in eachindex(ve_offset)]
        @test all(diff(ve_offset) .≈ Θ)

        # the offset only shifts the energies; the number of levels is fixed by the
        # cutoff, which is applied to the unshifted ladder E_i = (i + 1/2) Θ
        @test length(ve_offset) == length(ve_zero)
        @test length(ve_offset) == ceil(Int, E_DISS / Θ - 0.5)
        @test ve_offset[end] < E_DISS
        @test ve_offset[end] + Θ > E_DISS

        # having the same number of levels, the two ladders differ by exactly the
        # zero-point energy
        @test ve_offset .- ve_zero ≈ fill(0.5 * Θ, length(ve_zero))
    end

    @testset "harmonic ladder: cutoff uses the unshifted energies" begin
        # E_diss / Θ is chosen with fractional parts both below and above 1/2, so
        # that cutting on the unshifted ladder differs from cutting on the shifted
        # one; these constants distinguish the two rules. 76200 puts a level exactly
        # at the dissociation energy, which must be excluded
        for E_diss in (75840.0, E_DISS, 76200.0, 76800.0, 77000.0)
            ve_zero = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_diss;
                                                            ground_level_energy_zero = true)
            ve_offset = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_diss;
                                                              ground_level_energy_zero = false)

            # the level count never depends on where the ground level is put
            @test length(ve_zero) == length(ve_offset)

            # the last unshifted level is below the dissociation energy, the next is not
            n = length(ve_offset)
            @test (n - 0.5) * Θ < E_diss
            @test (n + 0.5) * Θ >= E_diss

            # the vanishing-anharmonicity limit of the anharmonic generator, which
            # cuts on the unshifted ladder by construction, must agree exactly
            @test ve_zero ≈ generate_e_vibr_arr_anharmonic_cutoff_K(Θ, 0.0, E_diss;
                                                                    ground_level_energy_zero = true)
            @test ve_offset ≈ generate_e_vibr_arr_anharmonic_cutoff_K(Θ, 0.0, E_diss;
                                                                      ground_level_energy_zero = false)
        end
    end

    @testset "anharmonic ladder" begin
        ve_harmonic = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS)
        ve_anharmonic = generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_A, E_DISS)

        @test eltype(ve_anharmonic) == Float64

        # anharmonicity lowers the levels, so more of them fit below dissociation
        @test length(ve_anharmonic) > length(ve_harmonic)
        @test ve_anharmonic[end] < E_DISS

        # the cutoff is applied to the unshifted ladder, so it has to be checked
        # there; the next level after the last one must reach the dissociation energy
        ve_unshifted = generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_A, E_DISS;
                                                               ground_level_energy_zero = false)
        n = length(ve_unshifted)
        @test ve_unshifted[end] < E_DISS
        @test (n + 0.5) * Θ - (n + 0.5)^2 * Θ_A >= E_DISS

        # levels get closer together as the level index grows, by 2 * Θ_anh per level
        @test all(diff(diff(ve_anharmonic)) .≈ -2 * Θ_A)

        # vanishing anharmonicity reproduces the harmonic ladder exactly
        ve_limit = generate_e_vibr_arr_anharmonic_cutoff_K(Θ, 0.0, E_DISS)
        @test length(ve_limit) == length(ve_harmonic)
        @test maximum(abs.(ve_limit - ve_harmonic)) < 2 * eps()
    end

    @testset "anharmonic ladder: ground level offset" begin
        ve_zero = generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_A, E_DISS;
                                                          ground_level_energy_zero = true)
        ve_offset = generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_A, E_DISS;
                                                            ground_level_energy_zero = false)

        @test ve_zero == generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_A, E_DISS)

        # unshifted levels follow E_i = (i + 1/2) Θ - (i + 1/2)^2 Θ_anh
        @test ve_offset ≈ [(i - 0.5) * Θ - (i - 0.5)^2 * Θ_A
                           for i in eachindex(ve_offset)]
        @test ve_offset[1] ≈ 0.5 * Θ - 0.25 * Θ_A

        # the cutoff is applied before shifting, so both flags give the same levels
        @test length(ve_zero) == length(ve_offset)
        @test ve_zero ≈ ve_offset .- ve_offset[1]
        @test ve_zero[1] == 0.0

        # repeated calls must not accumulate the shift
        @test generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_A, E_DISS;
                                                      ground_level_energy_zero = false) ≈
              ve_offset
    end

    @testset "Boltzmann averaging" begin
        ve = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS)

        # every level is equally populated in the infinite temperature limit
        # Z -> n_levels
        @test Z_vibr(ve, 1e12)≈length(ve) rtol=1e-6
        # only the ground level is populated as T -> 0
        # Z -> 1.0
        @test Z_vibr(ve, 1e-3) ≈ 1.0

        # averaging a constant returns that constant
        for T in TEMPERATURES
            @test avg_over_vibr_array(ve, fill(3.5, length(ve)), T) ≈ 3.5
        end

        # the average energy sits between the lowest and the highest level
        for T in TEMPERATURES
            avg = avg_over_vibr_array(ve, ve, T)
            @test ve[1] <= avg <= ve[end]
        end
    end

    @testset "ground level offset: energies and specific heats" begin
        # shifting every level by a constant multiplies all Boltzmann factors by a
        # common factor, which cancels in the partition function; the mean energy
        # therefore shifts by the same constant and the specific heat is unchanged
        ve = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS)

        for Δ in (0.5 * Θ, 12345.0)
            ve_shifted = ve .+ Δ
            for T in TEMPERATURES
                @test e_vibr_from_array(M, ve_shifted, T)≈e_vibr_from_array(M, ve, T) +
                                                          (k_B / M) * Δ rtol=1e-12
                @test c_vibr_from_array(M, ve_shifted, T)≈c_vibr_from_array(M, ve, T) rtol=1e-9
            end
        end

        # the same has to hold for the two ladders produced by the keyword, which
        # for these values contain the same number of levels
        ve_zero = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS;
                                                        ground_level_energy_zero = true)
        ve_offset = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS;
                                                          ground_level_energy_zero = false)
        @test length(ve_zero) == length(ve_offset)

        for T in TEMPERATURES
            @test e_vibr_from_array(M, ve_offset, T)≈e_vibr_from_array(M, ve_zero, T) +
                                                     (k_B / M) * 0.5 * Θ rtol=1e-12
            @test c_vibr_from_array(M, ve_offset, T)≈c_vibr_from_array(M, ve_zero, T) rtol=1e-9
        end

        # and for the anharmonic ladders, where both flags keep the same levels
        va_zero = generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_A, E_DISS;
                                                          ground_level_energy_zero = true)
        va_offset = generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_A, E_DISS;
                                                            ground_level_energy_zero = false)
        for T in TEMPERATURES
            @test e_vibr_from_array(M, va_offset, T)≈e_vibr_from_array(M, va_zero, T) +
                                                     (k_B / M) * va_offset[1] rtol=1e-12
            @test c_vibr_from_array(M, va_offset, T)≈c_vibr_from_array(M, va_zero, T) rtol=1e-9
        end
    end

    @testset "ground level offset: low temperature limit" begin
        T_cold = 50.0  # much smaller than Θ, so only the ground level is populated

        ve_zero = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS;
                                                        ground_level_energy_zero = true)
        ve_offset = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS;
                                                          ground_level_energy_zero = false)

        # with a zero-based ladder the vibrational energy vanishes,
        # with the zero-point offset it tends to (k_B / m) * Θ / 2
        @test e_vibr_from_array(M, ve_zero, T_cold) < 1e-8 * (k_B / M) * Θ
        @test e_vibr_from_array(M, ve_offset, T_cold)≈(k_B / M) * 0.5 * Θ rtol=1e-8

        # the specific heat is frozen out either way; it decays like
        # (Θ/T)^2 exp(-Θ/T), which the prefactor keeps well above exp(-Θ/T) itself
        @test c_vibr_from_array(M, ve_zero, T_cold) < 1e-6 * (k_B / M)
        @test c_vibr_from_array(M, ve_offset, T_cold) < 1e-6 * (k_B / M)
    end

    @testset "infinite harmonic oscillator" begin
        # test IHO by comparing to harmonic with very large E_diss
        ve_harmonic = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS)
        ve_harmonic_long1 = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS * 4)
        ve_harmonic_long2 = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS * 50)

        e_harmonic = e_vibr_from_array(M, ve_harmonic, T_MID)
        e_harmonic_long1 = e_vibr_from_array(M, ve_harmonic_long1, T_MID)
        e_harmonic_long2 = e_vibr_from_array(M, ve_harmonic_long2, T_MID)
        e_iho = e_vibr_iho(M, Θ, T_MID)

        # error should be becoming smaller
        @test abs(e_harmonic - e_iho) / e_iho > abs(e_harmonic_long1 - e_iho) / e_iho
        @test abs(e_harmonic_long1 - e_iho) / e_iho > abs(e_harmonic_long2 - e_iho) / e_iho

        # and become very small
        @test abs(e_harmonic_long2 - e_iho) / e_iho < 1e-14

        # the same for the specific heat
        @test c_vibr_from_array(M, ve_harmonic_long2, T_MID)≈c_vibr_iho(M, Θ, T_MID) rtol=1e-12

        # E_vibr - approximately kT at high temperatures
        @test abs(e_vibr_iho(M, Θ, T_HI) - k_B * T_HI / M) / e_vibr_iho(M, Θ, T_HI) < 0.025

        # c_vibr -> k_B / m at high temperatures, -> 0 at low ones
        @test c_vibr_iho(M, Θ, T_HI)≈k_B / M rtol=0.01
        @test c_vibr_iho(M, Θ, 50.0) < 1e-6 * (k_B / M)
    end

    @testset "infinite harmonic oscillator: ground level offset" begin
        # `e_vibr_iho` measures the energy from the ground level, so a ladder with
        # the zero-point offset exceeds it by exactly (k_B / m) * Θ / 2
        ve_offset = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS * 50;
                                                          ground_level_energy_zero = false)
        for T in TEMPERATURES
            @test e_vibr_from_array(M, ve_offset, T)≈e_vibr_iho(M, Θ, T) +
                                                     (k_B / M) * 0.5 * Θ rtol=1e-12
            @test c_vibr_from_array(M, ve_offset, T)≈c_vibr_iho(M, Θ, T) rtol=1e-9
        end
    end

    @testset "specific heat via finite differences" begin
        ladders = ("harmonic, zero-based" => generate_e_vibr_arr_harmonic_cutoff_K(Θ,
                                                                                   E_DISS),
                   "harmonic, offset" => generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS;
                                                                               ground_level_energy_zero = false),
                   "anharmonic, zero-based" => generate_e_vibr_arr_anharmonic_cutoff_K(Θ,
                                                                                       Θ_A,
                                                                                       E_DISS),
                   "anharmonic, offset" => generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_A,
                                                                                   E_DISS;
                                                                                   ground_level_energy_zero = false))

        for (name, ve) in ladders
            @testset "$name" begin
                for T in TEMPERATURES
                    fd = central_difference(t -> e_vibr_from_array(M, ve, t), T)
                    @test c_vibr_from_array(M, ve, T)≈fd rtol=1e-6
                end
            end
        end

        @testset "infinite harmonic oscillator" begin
            for T in TEMPERATURES
                fd = central_difference(t -> e_vibr_iho(M, Θ, t), T)
                @test c_vibr_iho(M, Θ, T)≈fd rtol=1e-6
            end
        end

        @testset "rotational" begin
            for T in TEMPERATURES
                fd = central_difference(t -> e_rot_cont(M, t), T)
                @test c_rot_cont(M, T)≈fd rtol=1e-12
            end
        end
    end

    @testset "rotational spectrum" begin
        for T in TEMPERATURES
            @test e_rot_cont(M, T) ≈ k_B * T / M
            @test c_rot_cont(M, T) ≈ k_B / M
        end
    end

    @testset "no internal degrees of freedom" begin
        @test e_c_none(M, T_LO, T_MID) == (0.0, 0.0)
    end

    @testset "generated energy/specific heat functions" begin
        ve = generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_DISS)

        generated = ("rotational" => (generate_e_c_rot_cont(),
                                      (m, T) -> e_rot_cont(m, T),
                                      (m, T) -> c_rot_cont(m, T)),
                     "vibrational, IHO" => (generate_e_c_vibr_iho(Θ),
                                            (m, T) -> e_vibr_iho(m, Θ, T),
                                            (m, T) -> c_vibr_iho(m, Θ, T)),
                     "vibrational, from array" => (generate_e_c_vibr_from_array(ve),
                                                   (m, T) -> e_vibr_from_array(m, ve, T),
                                                   (m, T) -> c_vibr_from_array(m, ve, T)))

        for (name, (f, e_ref, c_ref)) in generated
            @testset "$name" begin
                # the generators return a callable, not a value
                @test f isa Function

                for T in TEMPERATURES
                    # the contract `ThermoData1T` relies on: called with a mass and
                    # two temperatures, returning an indexable (e, c) pair
                    res = f(M, T, T)
                    @test res isa Tuple{Float64, Float64}
                    @test res[1] ≈ e_ref(M, T)
                    @test res[2] ≈ c_ref(M, T)
                end

                # energy and specific heat are evaluated at their own temperature,
                # which is what makes the pair usable for multi-temperature models
                e, c = f(M, T_LO, T_HI)
                @test e ≈ e_ref(M, T_LO)
                @test c ≈ c_ref(M, T_HI)
            end
        end
    end

    @testset "generated functions drive ThermoData1T" begin
        # reference quantities equal to one, i.e. scaling is the identity
        ref_q = ReferenceFlowQuantities(ntuple(_ -> 1.0, 14)...)

        T_min, T_max, ΔT = 100.0, 5000.0, 10.0
        td = ThermoData1T(ref_q, [M], [generate_e_c_vibr_iho(Θ)];
                          T_min = T_min, T_max = T_max, ΔT = ΔT)

        # at the tabulation points linear interpolation is exact, so the table has
        # to reproduce the translational contribution plus the generated one
        for T in (T_min, 1000.0, 2500.0, T_max - ΔT)
            index_e, frac_e, index_c, frac_c = get_index_lower_fracpos(T, td)
            @test frac_e≈0.0 atol=1e-12

            @test energy_component(1, index_e, frac_e, td)≈1.5 * k_B * T / M +
                                                           e_vibr_iho(M, Θ, T) rtol=1e-12
            @test c_v_component(1, index_c, frac_c, td)≈1.5 * k_B / M +
                                                        c_vibr_iho(M, Θ, T) rtol=1e-12
        end
    end
end

end # module
