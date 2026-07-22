@muladd begin
    # store interpolation tables
    # for internal energies and specific heats
    # assuming thermal equilibrium (one-temperature approximation)
    # NCOMP - number of flow components
    # energy values are stored in temperature range of [T_min_E, T_max_E] with step size ΔT
    # specific heat values are stored in temperature range of [T_min_c_v, T_max_c_v] with step size ΔT
    # species' atomic/molecular masses and their inverses are stored in the `mass` array
    # ALL quantities are stored in dimensionless, with scaling provided via a ReferenceFlowQuantities instance
    # T_tol sets the tolerance for the Newton solver for T(e)
    struct ThermoData1T{I<:Interpolation, CvO<:CvTableOffset, NCOMP} <: ThermoData
        ref_q::ReferenceFlowQuantities
        
        mass::SVector{NCOMP, Float64} # molecular mass of each component
        inv_mass::SVector{NCOMP, Float64} # inverse of molecular mass of each component

        T_min_E::Float64
        T_max_E::Float64

        T_min_c_v::Float64
        T_max_c_v::Float64

        ΔT::Float64
        inv_ΔT::Float64
        T_tol::Float64

        # N_cv_discretization is either N_T_discretization (NoCvOffset)
        # or N_T_discretization+2 (CvOffset)
        e_arr::Array{Float64, 2} # Species-specific tabulated energy, N_T_discretization*NCOMP
        c_v_arr::Array{Float64, 2} # Species-specific tabulated specific heat, N_cv_discretization*NCOMP or
        R_specific::SVector{NCOMP, Float64}  #  k_B / mass [J / kg / K] when not scaled ; component->R_specific

        e_min_arr::SVector{NCOMP, Float64}

        # temperatures at which e(T) is tabulated, T_min_E + i * ΔT
        T_arr::Vector{Float64}
        T_arr_inv::Vector{Float64}

        # temperatures at which c_v(T) is tabulated, T_min_c_v + i * ΔT
        # for NoCvOffset these alias T_arr / T_arr_inv
        T_c_arr::Vector{Float64}
        T_c_arr_inv::Vector{Float64}

        # used to estimate \int c_v(tau) / tau d tau
        int_c_v_over_t_arr::Array{Float64,2} # Species-specific tabulated integral part of entropy, N_cv_discretization*NCOMP

        # construct a `ThermoData1T` instance for a flow with NCOMP species
        # using a ReferenceFlowQuantities instance for scaling
        # `mass_arr` is the array of the species' molecular/atomic masses in kg
        # `e_c_int_function_arr` is an array of NCOMP tuples/arrays of functions
        # each of which computes the specific internal energy and specific heat
        # given a value of the species' mass and two temperatures at which to compute
        # the energy and specific heat
        # the energy and specific heats of the translational degrees of freedom are
        # account for inside the code and should not be part of the computations in
        # e_c_int_function_arr
        function ThermoData1T{LinearInterpolation, NoCvOffset, NCOMP}(ref_q, mass_arr, e_c_int_function_arr;
                                                                      T_min=10.0, T_max=3.0e4, 
                                                                      T_tol = 1e-9, ΔT=1.0) where {NCOMP}
                @assert (length(mass_arr)==length(e_c_int_function_arr)==NCOMP)

                n_T = trunc(Int, (T_max - T_min) / ΔT) + 1
                T_arr = Vector(LinRange(T_min, T_max, n_T))
                @assert abs((T_arr[2] - T_arr[1]) - ΔT) < 1e-3

                inv_ΔT = 1.0 / ΔT

                # Preallocate some arrays
                e_arr = zeros(n_T, NCOMP)
                c_v_arr = zeros(n_T, NCOMP)
                R_specific = zeros(NCOMP)
                e_min_arr = zeros(NCOMP)
                int_c_v_over_t_arr = zeros(n_T, NCOMP)

                e_arr = map((m, f)->map(t -> 3.0/2.0 * k_B * t / m .+ f(m, t, t)[1], T_arr), mass_arr, e_c_int_function_arr)
                e_arr = transpose(stack(e_arr, dims=1))
                e_min_arr = vec(minimum(e_arr, dims=1)) ./ ref_q.e_ref
                c_v_arr = transpose(stack(map((m, f)->(map(t-> f(m, t, t)[2], T_arr) .+ ((3.0 / 2.0) * k_B / m)), mass_arr, e_c_int_function_arr), dims=1))

                @inbounds for j in 1:NCOMP
                    # Integrate ∫ c_v / T dT
                    for i in 2:n_T
                        T_a, T_b = T_arr[i-1], T_arr[i]
                        c_v_a, c_v_b = c_v_arr[i-1, j], c_v_arr[i, j]

                        int_c_v_over_t_arr[i, j] = int_c_v_over_t_arr[i-1, j] +
                                                (c_v_a - (c_v_b - c_v_a) * T_a / ΔT) * log(T_b / T_a) + (c_v_b - c_v_a)
                    end
                end

                T_min /= ref_q.T_ref
                T_max /= ref_q.T_ref
                ΔT /= ref_q.T_ref
                T_arr ./= ref_q.T_ref
                T_arr_inv = 1.0 ./ T_arr

                e_arr ./= ref_q.e_ref
                c_v_arr ./= ref_q.c_v_ref
                int_c_v_over_t_arr ./= ref_q.c_v_ref
                mass_arr = mass_arr ./ ref_q.m_ref

                return new(ref_q,
                        mass_arr,
                        1.0 ./ mass_arr,
                        T_min, T_max,
                        T_min, T_max,
                        ΔT, 1.0 / ΔT, T_tol,
                        e_arr, c_v_arr,
                        (k_B ./ mass_arr) ./ ref_q.c_v_ref,
                        e_min_arr,
                        T_arr, T_arr_inv,
                        T_arr, T_arr_inv,  # c_v tabulated on the same grid as e
                        int_c_v_over_t_arr)
        end

        # same as above, but c_v (and ∫ c_v / T dT) are tabulated on a grid shifted by -ΔT/2
        # w.r.t. the energy grid: T_c_arr[i] = T_min - ΔT/2 + (i - 1) * ΔT, i = 1 ... n_T + 2
        # the two extra points guarantee that [T_min, T_max] is strictly bracketed by the c_v grid
        function ThermoData1T{LinearInterpolation, CvOffset, NCOMP}(ref_q, mass_arr, e_c_int_function_arr;
                                                                     T_min=10.0, T_max=3.0e4,
                                                                     T_tol = 1e-9, ΔT=1.0) where {NCOMP}
                @assert (length(mass_arr)==length(e_c_int_function_arr)==NCOMP)
                # the c_v grid starts at T_min - ΔT/2, which has to stay positive for ∫ c_v / T dT
                @assert T_min > 0.5 * ΔT

                n_T = trunc(Int, (T_max - T_min) / ΔT) + 1
                T_arr = Vector(LinRange(T_min, T_max, n_T))
                @assert abs((T_arr[2] - T_arr[1]) - ΔT) < 1e-3

                n_c = n_T + 2
                T_c_min = T_min - 0.5 * ΔT
                T_c_max = T_c_min + (n_c - 1) * ΔT
                T_c_arr = Vector(LinRange(T_c_min, T_c_max, n_c))

                inv_ΔT = 1.0 / ΔT

                # Preallocate some arrays
                e_arr = zeros(n_T, NCOMP)
                c_v_arr = zeros(n_c, NCOMP)
                R_specific = zeros(NCOMP)
                e_min_arr = zeros(NCOMP)
                int_c_v_over_t_arr = zeros(n_c, NCOMP)

                e_arr = map((m, f)->map(t -> 3.0/2.0 * k_B * t / m .+ f(m, t, t)[1], T_arr), mass_arr, e_c_int_function_arr)
                e_arr = transpose(stack(e_arr, dims=1))
                e_min_arr = vec(minimum(e_arr, dims=1)) ./ ref_q.e_ref
                c_v_arr = transpose(stack(map((m, f)->(map(t-> f(m, t, t)[2], T_c_arr) .+ ((3.0 / 2.0) * k_B / m)), mass_arr, e_c_int_function_arr), dims=1))

                @inbounds for j in 1:NCOMP
                    # Integrate ∫ c_v / T dT on the c_v grid
                    for i in 2:n_c
                        T_a, T_b = T_c_arr[i-1], T_c_arr[i]
                        c_v_a, c_v_b = c_v_arr[i-1, j], c_v_arr[i, j]

                        int_c_v_over_t_arr[i, j] = int_c_v_over_t_arr[i-1, j] +
                                                (c_v_a - (c_v_b - c_v_a) * T_a / ΔT) * log(T_b / T_a) + (c_v_b - c_v_a)
                    end
                end

                T_min /= ref_q.T_ref
                T_max /= ref_q.T_ref
                T_c_min /= ref_q.T_ref
                T_c_max /= ref_q.T_ref
                ΔT /= ref_q.T_ref
                T_arr ./= ref_q.T_ref
                T_c_arr ./= ref_q.T_ref

                e_arr ./= ref_q.e_ref
                c_v_arr ./= ref_q.c_v_ref
                int_c_v_over_t_arr ./= ref_q.c_v_ref
                mass_arr ./= ref_q.m_ref

                return new(ref_q,
                        mass_arr,
                        1.0 ./ mass_arr,
                        T_min, T_max,
                        T_c_min, T_c_max,
                        ΔT, 1.0 / ΔT, T_tol,
                        e_arr, c_v_arr,
                        (k_B ./ mass_arr) ./ ref_q.c_v_ref,
                        e_min_arr,
                        T_arr, 1.0 ./ T_arr,
                        T_c_arr, 1.0 ./ T_c_arr,
                        int_c_v_over_t_arr)
        end

        function ThermoData1T(ref_q::ReferenceFlowQuantities, mass_arr, e_c_int_function_arr;
            T_min=10.0, T_max=3.0e4, T_tol=1e-9, ΔT=1.0, interpolation=:linear, cv_table_offset=false)
            NCOMP = length(mass_arr)
            if interpolation == :linear
                if cv_table_offset == true
                    return ThermoData1T{LinearInterpolation, CvOffset, NCOMP}(ref_q, mass_arr, e_c_int_function_arr; T_min=T_min, T_max=T_max, T_tol=T_tol, ΔT=ΔT)
                else
                    return ThermoData1T{LinearInterpolation, NoCvOffset, NCOMP}(ref_q, mass_arr, e_c_int_function_arr; T_min=T_min, T_max=T_max, T_tol=T_tol, ΔT=ΔT)
                end
            else
                error("Non-linear interpolation not implemented")
            end
        end
    end

    @inline function ncomponents(thermodata::ThermoData1T{I, CvO, NCOMP}) where {I, CvO, NCOMP}
        NCOMP
    end

    @inline function eachcomponent(thermodata::ThermoData1T)
        Base.OneTo(ncomponents(thermodata))
    end

    # return index and fractional position for energy and cv interpolation in case of no offset
    @inline function get_index_lower_fracpos(T, thermodata::ThermoData1T{I, NoCvOffset, NCOMP}) where {I, NCOMP}
        fracpos = (T - thermodata.T_min_E) * thermodata.inv_ΔT
        index_lower = floor(Int, fracpos)
        fracpos -= index_lower
        index_lower += 1

        return index_lower, fracpos, index_lower, fracpos
    end

    # return index and fractional position for energy and cv interpolation in case the
    # c_v table is offset by -ΔT/2 w.r.t. the energy table
    # the position on the c_v grid is (T - (T_min_E - ΔT/2)) / ΔT = fracpos_global + 1/2,
    # so it only differs from the energy one by a constant shift of 1/2 and can be obtained
    # from `fracpos` with a single comparison - no second division/floor needed
    @inline function get_index_lower_fracpos(T, thermodata::ThermoData1T{I, CvOffset, NCOMP}) where {I, NCOMP}
        fracpos = (T - thermodata.T_min_E) * thermodata.inv_ΔT
        index_lower = floor(Int, fracpos)
        fracpos -= index_lower
        index_lower += 1

        # fracpos + 1/2 lies in [1/2, 3/2), so its integer part is exactly `shift`
        shift = fracpos >= 0.5
        index_lower_c = index_lower + shift
        fracpos_c = fracpos + 0.5 - shift

        return index_lower, fracpos, index_lower_c, fracpos_c
    end

    # compute energy of species i_comp using linear interpolation
    @inline function energy_component(i_comp, index_lower_e, fracpos_e, thermodata::ThermoData1T{LinearInterpolation, CvO, NCOMP}) where {CvO, NCOMP}
        @inbounds return thermodata.e_arr[index_lower_e, i_comp] * (1.0 - fracpos_e) + fracpos_e * thermodata.e_arr[index_lower_e + 1, i_comp]
    end

    # compute energy given temperature and array of densities
    # used for prim2cons transformation
    @inline function energy_from_rho_vec(rho_vec::SVector, rho, T, thermodata::ThermoData1T{LinearInterpolation, CvO, NCOMP}) where {CvO, NCOMP}
        result = 0.0
        
        index_lower_e, fracpos_e, _, _ = get_index_lower_fracpos(T, thermodata)
        @inbounds for i in eachcomponent(thermodata)
            result += energy_component(i, index_lower_e, fracpos_e, thermodata) * rho_vec[i]
        end
        
        return result / rho
    end

    @inline function c_v_component(i_comp, index_lower_c, fracpos_c, thermodata::ThermoData1T{LinearInterpolation, CvO, NCOMP}) where {CvO, NCOMP}
        @inbounds return thermodata.c_v_arr[index_lower_c, i_comp] * (1.0 - fracpos_c) + fracpos_c * thermodata.c_v_arr[index_lower_c + 1, i_comp]
    end

    # compute specific energy of flow given values of interpolation point and fractional position 
    # rho_inv = 1/rho
    @inline function energy(u, rho_inv, index_lower_e, fracpos_e, thermodata::ThermoData1T{I, CvO, NCOMP}) where {I, CvO, NCOMP}
        result = 0.0
        @inbounds for i in eachcomponent(thermodata)
            result += energy_component(i, index_lower_e, fracpos_e, thermodata) * u[i + 3]
        end
        return result * rho_inv
    end

    # compute specific heat capacity of flow given values of interpolation point and fractional position 
    # rho_inv = 1/rho
    @inline function c_v(u, rho_inv, index_lower_c, fracpos_c, thermodata::ThermoData1T{I, CvO, NCOMP}) where {I, CvO, NCOMP}
        result = 0.0
        @inbounds for i in eachcomponent(thermodata)
            result += c_v_component(i, index_lower_c, fracpos_c, thermodata) * u[i + 3]
        end
        return result * rho_inv
    end

    # compute ∫ c_v / T dT for a single component using linear interpolation
    @inline function entropy_c_v_integral_component(i_comp, index_lower_c, fracpos_c, T_b, thermodata::ThermoData1T{LinearInterpolation, CvO, NCOMP}) where {CvO, NCOMP}
        @inbounds T_a = thermodata.T_c_arr[index_lower_c]
        @inbounds T_a_inv = thermodata.T_c_arr_inv[index_lower_c]

        c_v_b = c_v_component(i_comp, index_lower_c, fracpos_c, thermodata)  # value of c_v at T
        @inbounds c_v_a = thermodata.c_v_arr[index_lower_c, i_comp]    # value of c_v at closest_T
        integrate_part = (c_v_a - (c_v_b - c_v_a) * T_a * thermodata.inv_ΔT) * log(T_b * T_a_inv) + (c_v_b - c_v_a)

        @inbounds return thermodata.int_c_v_over_t_arr[index_lower_c, i_comp] + integrate_part
    end

    # compute ∫ c_v / T dT of flow
    @inline function entropy_c_v_integral(u, T, rho, thermodata::ThermoData1T)
        result = 0.0

        _, _, index_lower_c, fracpos_c = get_index_lower_fracpos(T, thermodata)
        @inbounds for i in eachcomponent(thermodata)
            result += entropy_c_v_integral_component(i, index_lower_c, fracpos_c, T, thermodata) * u[i + 3] / rho
        end
        return result
    end

    # check if energy is too low, return true if it is
    # u is vector of conservative flow variables,
    # rho_inv is 1/rho, e is the internal energy per unit mass
    @inline function limit_T_low_rho_inv(u, rho_inv, e, thermodata::ThermoData1T)
        e_min = 0.0
        @inbounds for i in eachcomponent(thermodata)
            e_min += thermodata.e_min_arr[i] * u[i + 3]
        end
        e_min *= rho_inv
        if e <= e_min
            return true
        else
            return false
        end
    end

    # compute T(e) via Newton iteration, clamping T to be in range of [1.0001 * T_min_E, 0.9999 * thermodata.T_max_E]
    # and return T, index_lower_e, fracpos_e, index_lower_c, fracpos_c (for interpolation)
    # u is vector of conservative flow variables,
    # rho_inv is 1/rho, T0 is the initial guess for T, e is the internal energy per unit mass
    @inline function temperature_rho_inv_with_index(u, rho_inv, T0, e, thermodata::ThermoData1T)
        T = T0

        if (T < thermodata.T_min_E)
            return (1.0001 * thermodata.T_min_E, get_index_lower_fracpos(1.0001 * thermodata.T_min_E, thermodata)...)
        elseif (T > thermodata.T_max_E)
            return (0.9999 * thermodata.T_max_E, get_index_lower_fracpos(0.9999 * thermodata.T_max_E, thermodata)...)
        end

        if limit_T_low_rho_inv(u, rho_inv, e, thermodata)
            return (1.0001 * thermodata.T_min_E, get_index_lower_fracpos(1.0001 * thermodata.T_min_E, thermodata)...)
        end

        index_lower_e, fracpos_e, index_lower_c, fracpos_c = get_index_lower_fracpos(T, thermodata)
        fx = energy(u, rho_inv, index_lower_e, fracpos_e, thermodata) - e

        mintol = thermodata.T_tol * e + thermodata.T_tol
        
        while abs(fx) > mintol
            T -= fx / c_v(u, rho_inv, index_lower_c, fracpos_c, thermodata)
            index_lower_e, fracpos_e, index_lower_c, fracpos_c = get_index_lower_fracpos(T, thermodata)
            fx = energy(u, rho_inv, index_lower_e, fracpos_e, thermodata) - e
            # iter += 1
        end
        return T, index_lower_e, fracpos_e, index_lower_c, fracpos_c
    end

    # compute T(e) via Newton iteration, clamping T to be in range of [1.0001 * T_min_E, 0.9999 * thermodata.T_max_E]
    # and return T
    # u is vector of conservative flow variables,
    # rho_inv is 1/rho, T0 is the initial guess for T, e is the internal energy per unit mass
    @inline function temperature_rho_inv(u, rho_inv, T0, e, thermodata::ThermoData1T)
        T = T0

        if (T < thermodata.T_min_E)
            return 1.0001 * thermodata.T_min_E
        elseif (T > thermodata.T_max_E)
            return 0.9999 * thermodata.T_max_E
        end

        if limit_T_low_rho_inv(u, rho_inv, e, thermodata)
            return 1.0001 * thermodata.T_min_E
        end

        index_lower_e, fracpos_e, index_lower_c, fracpos_c = get_index_lower_fracpos(T, thermodata)
        fx = energy(u, rho_inv, index_lower_e, fracpos_e, thermodata) - e

        mintol = thermodata.T_tol * e + thermodata.T_tol
        
        while abs(fx) > mintol
            T -= fx / c_v(u, rho_inv, index_lower_c, fracpos_c, thermodata)
            index_lower_e, fracpos_e, index_lower_c, fracpos_c = get_index_lower_fracpos(T, thermodata)
            fx = energy(u, rho_inv, index_lower_e, fracpos_e, thermodata) - e
            # iter += 1
        end
        return T
    end

    # compute adiabatic index γ(T) via interpolation
    # rho_inv = 1/rho
    @inline function get_gamma(u, rho_inv, index_lower_c, fracpos_c, thermodata::ThermoData1T)
        c_v_val = c_v(u, rho_inv, index_lower_c, fracpos_c, thermodata)
        c_p = 0.0

        # c_p = c_v + \sum_i rho_i k/m_i/rho =(scaling)= \sum_i rho_i' 1.0/m_i'/rho' (' denotes scaled variables)
        @inbounds for i in eachcomponent(thermodata)
            c_p += u[i + 3] * thermodata.inv_mass[i]
        end

        return (c_v_val + c_p * rho_inv) / c_v_val
    end
end # @muladd
