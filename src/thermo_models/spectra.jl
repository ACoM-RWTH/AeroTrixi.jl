# This file contains functions to compute rotational and vibrational specific energies
# and specific heats, as well as to compute vibrational energy spectra for different
# models (cut-off harmonic oscillator, cut-off anharmonic oscillator). The latter
# can be used in the _from_array functions to compute vibrational specific energies
# and specific heats by averaging the vibrational spectrum.

# compute specific rotational energy and specific heat of internal degrees of freedom
# assuming no internal degrees of freedom, i.e. return 0
function e_c_none(m, T)
    return 0.0, 0.0
end

"""
    e_rot_cont(m, T)

Specific rotational energy (J/kg) of a particle of mass `m` (kg) at temperature `T` (K),
for a continuous, fully excited spectrum.
"""
function e_rot_cont(m, T)
    return k_B * T / m
end

"""
    c_rot_cont(m, T)

Rotational specific heat (J/(kg K)) for a continuous, fully excited spectrum; see
[`e_rot_cont`](@ref).
"""
function c_rot_cont(m, T)
    return k_B / m
end

"""
    generate_e_c_rot_cont()

Function `(m, T) -> (e, c)` with the rotational energy and specific heat of
[`e_rot_cont`](@ref) and [`c_rot_cont`](@ref).
"""
function generate_e_c_rot_cont()
    return (m, T) -> (e_rot_cont(m, T), c_rot_cont(m, T))
end

"""
    e_vibr_iho(m, Θ, Tv)

Specific vibrational energy (J/kg) of the infinite harmonic oscillator with the
characteristic temperature `Θ` (K) at the vibrational temperature `Tv` (K).
"""
function e_vibr_iho(m, Θ, Tv)
    return (k_B / m) * Θ / (exp(Θ / Tv) - 1.0)
end

"""
    c_vibr_iho(m, Θ, Tv)

Vibrational specific heat (J/(kg K)) of the infinite harmonic oscillator; see
[`e_vibr_iho`](@ref).
"""
function c_vibr_iho(m, Θ, Tv)
    return (k_B / m) * (Θ / Tv)^2 * exp(Θ / Tv) / (exp(Θ / Tv) - 1.0)^2
end

"""
    generate_e_c_vibr_iho(Θ)

Function `(m, Tv) -> (e, c)` with the vibrational energy and specific heat of
[`e_vibr_iho`](@ref) and [`c_vibr_iho`](@ref) for the characteristic temperature `Θ`.
"""
function generate_e_c_vibr_iho(Θ)
    return (m, Tv) -> (e_vibr_iho(m, Θ, Tv), c_vibr_iho(m, Θ, Tv))
end

"""
    generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_diss; ground_level_energy_zero = true)

Vibrational level energies (K) of the harmonic oscillator, ``E_i = (i + 1/2) Θ``, below the
dissociation energy `E_diss` (K); shifted to ``E_0 = 0`` if `ground_level_energy_zero`.
"""
function generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_diss; ground_level_energy_zero = true)
    offset = ground_level_energy_zero ? 0.0 : 0.5
    i_max = ceil(Int, (E_diss / Θ - 0.5)) - 1  # find maximum level before vibrational energy reaches dissociation energy
    # this is with respect to the energy ladder with non-zero ground level energy
    # a level whose energy is exactly E_diss is not included, as in the anharmonic case
    return (collect(0:i_max) .+ offset) .* Θ  # units are K
end

"""
    generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_anh, E_diss; ground_level_energy_zero = true)

Anharmonic level energies (K), ``E_i = (i + 1/2) Θ - (i + 1/2)^2 Θ_{anh}``, below `E_diss`
(K); see [`generate_e_vibr_arr_harmonic_cutoff_K`](@ref).
"""
function generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_anh, E_diss;
                                                 ground_level_energy_zero = true)
    v_e_arr = Float64[]
    i = 0
    while (true)
        # compute vibrational energy of state i
        # if it's less than dissociation energy, add it to the array
        # if it's larger, stop computation
        enew = (i + 0.5) * Θ - (i + 0.5)^2 * Θ_anh
        if (enew < E_diss)
            push!(v_e_arr, enew)
            i += 1
        else
            break
        end
    end
    if ground_level_energy_zero
        return v_e_arr .- v_e_arr[1]
    else
        return v_e_arr
    end
end

# compute vibrational partition function given an array of vibrational energies
# assuming a Boltzmann distribution of the vibrational energies
function Z_vibr(E_vibr_array_K, Tv)
    return sum(exp.(-E_vibr_array_K / Tv))
end

# average a quantity dependent on vibrational energy over vibrational spectrum 
# quantity to average passed as an array (computed for each vibrational level)
# assuming a Boltzmann distribution of the vibrational energies
function avg_over_vibr_array(E_vibr_array_K, array_to_avg, Tv)
    Z_v = Z_vibr(E_vibr_array_K, Tv)

    return sum(array_to_avg .* exp.(-E_vibr_array_K / Tv)) / Z_v
end

"""
    e_vibr_from_array(m, E_vibr_array_K, Tv)

Specific vibrational energy (J/kg) from the level energies `E_vibr_array_K` (K), averaged
with a Boltzmann distribution at the vibrational temperature `Tv` (K).
"""
function e_vibr_from_array(m, E_vibr_array_K, Tv)
    return (k_B / m) * avg_over_vibr_array(E_vibr_array_K, E_vibr_array_K, Tv)
end

"""
    c_vibr_from_array(m, E_vibr_array_K, Tv)

Vibrational specific heat (J/(kg K)) from the level energies `E_vibr_array_K` (K) with a
Boltzmann distribution; see [`e_vibr_from_array`](@ref).
"""
function c_vibr_from_array(m, E_vibr_array_K, Tv)
    avg_e_sq = avg_over_vibr_array(E_vibr_array_K, E_vibr_array_K .^ 2, Tv)
    avg_e = avg_over_vibr_array(E_vibr_array_K, E_vibr_array_K, Tv)
    return (k_B / m) * (avg_e_sq - avg_e^2) / Tv^2
end

"""
    generate_e_c_vibr_from_array(E_vibr_array_K)

Function `(m, Tv) -> (e, c)` with the vibrational energy and specific heat of
[`e_vibr_from_array`](@ref) and [`c_vibr_from_array`](@ref) for the given level energies.
"""
function generate_e_c_vibr_from_array(E_vibr_array_K)
    return (m, Tv) -> (e_vibr_from_array(m, E_vibr_array_K, Tv),
                       c_vibr_from_array(m, E_vibr_array_K, Tv))
end
