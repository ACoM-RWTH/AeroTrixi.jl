function e_c_none(m, T_e_rot, T_c_rot)
    return 0.0, 0.0
end

# compute specific rotational energy assuming a continuous and fully excited spectrum
# [e] = J / kg
function e_rot_cont(m, T)
    return k_B * T / m
end

# compute specific heat of rotational degrees of freedom assuming a continuous and fully excited spectrum
# [c] = J / kg / K
function c_rot_cont(m, T)
    return k_B / m
end

# returns a function computing the specific rotational energy and specific heat of rotational degrees of freedom assuming a continuous and fully excited spectrum
# [e] = J / kg, [c] = J / kg / K
function generate_e_c_rot_cont()
    return (m, T_e_rot, T_c_rot) -> (e_rot_cont(m, T_e_rot), c_rot_cont(m, T_c_rot))
end

# compute specific vibrational energy using the infinite harmonic oscillator model
# [e] = J / kg
function e_vibr_iho(m, Θ, T)
    return (k_B / m) * Θ ./ ((exp.(Θ./T) .- 1.0))
end

# compute specific heat of vibrational degrees of freedom using the infinite harmonic oscillator model
# [c] = J / kg / K
function c_vibr_iho(m, Θ, T)
    return (k_B / m) * (Θ ./ T).^2 .* exp.(Θ ./ T) ./ (exp.(Θ ./ T) .- 1.0).^2
end

# returns a function computing the specific vibrational energy and specific heat of vibrational degrees of freedom using the infinite harmonic oscillator model
# [e] = J / kg, [c] = J / kg / K
function generate_e_c_vibr_iho(Θ)
    return (m, T_e_vibr, T_c_vibr) -> (e_vibr_iho(m, Θ, T_e_vibr), c_vibr_iho(m, Θ, T_c_vibr))
end

# generate list of vibrational energies whose energy does not exceed dissociation energy
# assuming a harmonic oscillator model with characteristic vibrational temperature Θ
# if ground_level_energy_zero == false, E_i = (i + 0.5) * Θ, i = 0, ..., i_max
# if ground_level_energy_zero == true, E_i = i * Θ, i = 0, 1, 2, ..., i_max
# units of computed array are K
function generate_e_vibr_arr_harmonic_cutoff_K(Θ, E_diss; ground_level_energy_zero = true)
    offset = ground_level_energy_zero ? 0.0 : 0.5
    i_max = ceil(Int, (E_diss / Θ - 0.5)) - 1  # find maximum level before vibrational energy reaches dissociation energy
    # this is with respect to the energy ladder with non-zero ground level energy
    # a level whose energy is exactly E_diss is not included, as in the anharmonic case
    return (collect(0:i_max) .+ offset) .* Θ  # units are K
end

# generate list of vibrational energies whose energy does not exceed dissociation energy
# assuming an anharmonic oscillator model with characteristic vibrational temperature Θ
# and anharmonic factor Θ_anh
# if ground_level_energy_zero == false, E_i = (i + 0.5) * Θ - (i + 0.5)^2 * Θ_anh, i = 0, ..., i_max
# if ground_level_energy_zero == true, E_i(ground_level_energy_zero = false) - E_0(ground_level_energy_zero = false)
# units of computed array are K
function generate_e_vibr_arr_anharmonic_cutoff_K(Θ, Θ_anh, E_diss; ground_level_energy_zero = true)
    v_e_arr = Float64[]
    i = 0
    while(true)
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
function Z_vibr(E_vibr_array_K, T)
    return sum(exp.(-E_vibr_array_K ./ T))
end

# average a quantity dependent on vibrational energy over vibrational spectrum 
# quantity to average passed as an array (computed for each vibrational level)
# assuming a Boltzmann distribution of the vibrational energies
function avg_over_vibr_array(E_vibr_array_K, array_to_avg, T)
    Z_v = Z_vibr(E_vibr_array_K, T)

    return sum(array_to_avg .* exp.(-E_vibr_array_K ./ T)) / Z_v
end

# compute specific vibrational energy using a cut-off model by averaging
# vibrational energies of the vibrational levels over the vibrational spectrum
# assuming a Boltzmann distribution of the vibrational energies
# [e] = J / kg
function e_vibr_from_array(m, E_vibr_array_K, T)
    return (k_B / m) * avg_over_vibr_array(E_vibr_array_K, E_vibr_array_K, T)
end

# compute specific heat of vibrational degrees of freedom using a cut-off model by averaging
# vibrational energies of the vibrational levels over the vibrational spectrum
# assuming a Boltzmann distribution of the vibrational energies
# [c] = J / kg / K
function c_vibr_from_array(m, E_vibr_array_K, T)
    avg_e_sq = avg_over_vibr_array(E_vibr_array_K, E_vibr_array_K .^ 2, T) 
    avg_e = avg_over_vibr_array(E_vibr_array_K, E_vibr_array_K, T)
    return (k_B / m) * (avg_e_sq - avg_e^2) / T^2
end

# returns a function computing the specific vibrational energy
# and specific heat of vibrational degrees of freedom using a cut-off model by averaging
# vibrational energies of the vibrational levels over the vibrational spectrum
# assuming a Boltzmann distribution of the vibrational energies
# [e] = J / kg, [c] = J / kg / K
function generate_e_c_vibr_from_array(E_vibr_array_K)
    return (m, T_e_vibr, T_c_vibr) -> (e_vibr_from_array(m, E_vibr_array_K, T_e_vibr), c_vibr_from_array(m, E_vibr_array_K, T_c_vibr))
end

