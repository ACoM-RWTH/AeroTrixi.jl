@muladd begin
    # compressible Euler equations for 2D multi-species flows with 1 temperature
    # (thermal equilibrium)
    # with interpolation for thermodynamic data stored in a ThermoData1T{I, CvO, NCOMP} instance
    # the conservative variables are rho v_1, rho v_2, rho e, rho_1, ..., rho_NCOMP
    # the primitive variables are v_1, v_2, T, rho_1, ..., rho_NCOMP
    struct CompressibleEulerEquationsMs1T2D{I, CvO, NVARS, NCOMP} <: 
        AbstractCompressibleEulerMulticomponentEquations{2, NVARS, NCOMP}
        # NVARS = NCOMP + 3 (NCOMP eqns for density + 2 for velocity and 1 for energy)

        min_T_jump::Float64
        thermodata::ThermoData1T{I, CvO, NCOMP}

        # Inner constructor
        function CompressibleEulerEquationsMs1T2D{I, CvO, NVARS, NCOMP}(
            ref_q::ReferenceFlowQuantities, mass_arr, e_int_function, c_int_function;
            T_min=10.0, T_max=3.0e4, T_tol=1e-9, ΔT=1.0, min_T_jump=1e-6
        ) where {I, CvO, NVARS, NCOMP}
            e_c_int_function_arr = map((e_fun, c_fun) -> ((m, T, _) -> (e_fun(T), c_fun(T))),
                                    e_int_function, c_int_function)

            thermodata = ThermoData1T{I, CvO, NCOMP}(
                ref_q, mass_arr, e_c_int_function_arr;
                T_min=T_min, T_max=T_max, T_tol=T_tol, ΔT=ΔT
            )
            ΔT /= ref_q.T_ref

            return new{I, CvO, NVARS, NCOMP}(min_T_jump * ΔT, thermodata)
        end

        # Outer constructor (user-friendly)
        function CompressibleEulerEquationsMs1T2D(ref_q::ReferenceFlowQuantities,
                                                mass_arr, e_int_function, c_int_function;
                                                T_min=10.0, T_max=3.0e4, T_tol=1e-9, ΔT=1.0, min_T_jump=1e-6,
                                                interpolation=:linear, cv_table_offset=false)

            NCOMP = length(mass_arr)
            NVARS = NCOMP + 3

            if interpolation == :linear
                if cv_table_offset == true
                    return CompressibleEulerEquationsMs1T2D{LinearInterpolation, CvOffset, NVARS, NCOMP}(
                        ref_q, mass_arr, e_int_function, c_int_function;
                        min_T_jump=min_T_jump, T_min=T_min, T_max=T_max, T_tol=T_tol, ΔT=ΔT
                    )
                else
                    return CompressibleEulerEquationsMs1T2D{LinearInterpolation, NoCvOffset, NVARS, NCOMP}(
                        ref_q, mass_arr, e_int_function, c_int_function;
                        min_T_jump=min_T_jump, T_min=T_min, T_max=T_max, T_tol=T_tol, ΔT=ΔT
                    )
                end
            else
                error("Non-linear interpolation not implemented")
            end
        end
    end

    @inline function density(u, equations::CompressibleEulerEquationsMs1T2D)
        @inbounds rho = zero(u[1])

        @inbounds for i in eachcomponent(equations.thermodata)
            rho += u[i + 3]
        end

        return rho
    end

    @inline function densities(u, v, equations::CompressibleEulerEquationsMs1T2D)
        @inbounds return SVector{ncomponents(equations), real(equations)}(u[i + 3] * v
                                                                          for i in eachcomponent(equations))
    end

    # Calculate total energy for a conservative state `cons`
    @inline energy_total(cons, ::CompressibleEulerEquationsMs1T2D) = cons[3]

    # Calculate kinetic energy for a conservative state `cons`
    @inline function energy_kinetic(u, rho, equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, _ = u
        return (rho_v1^2 + rho_v2^2) / (2 * rho)
    end

    @inline function energy_kinetic(u, equations::CompressibleEulerEquationsMs1T2D)
        rho = density(u, equations)
        return energy_kinetic(u, rho, equations)#(rho_v1^2 + rho_v2^2) / (2 * rho)
    end

    @inline function energy_internal(cons, rho, equations::CompressibleEulerEquationsMs1T2D)
        # this returns rho e_internal [J/m^3]
        return energy_total(cons, equations) - energy_kinetic(cons, rho, equations)
    end

    @inline function energy_internal(cons, equations::CompressibleEulerEquationsMs1T2D)
        # this returns rho e_internal [J/m^3]
        return energy_total(cons, equations) - energy_kinetic(cons, equations)
    end

    # compute temperature
    @inline function temperature(u, equations::CompressibleEulerEquationsMs1T2D)
        rho = density(u, equations)
        rho_inv = 1.0 / rho
        eint = energy_internal(u, equations) * rho_inv
        return temperature_rho_inv(u, rho_inv, 0.28*eint, eint, equations.thermodata)
    end

    # compute total density and number density 
    @inline function density_and_number_density(u, equations::CompressibleEulerEquationsMs1T2D)
        rho = zero(u[1])
        nrho = zero(u[1])

        @inbounds for i in eachcomponent(equations.thermodata)
            rho += u[i + 3]
            nrho += u[i + 3] * equations.thermodata.inv_mass[i]
        end

        return rho, nrho
    end

    # compute pressure
    @inline function pressure(u, equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e, _ = u
        rho, nrho = density_and_number_density(u, equations)
        rho_inv = 1.0 / rho
        e_internal = (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) * rho_inv) * rho_inv
        p = temperature_rho_inv(u, rho_inv, 0.28*e_internal, e_internal, equations.thermodata) * nrho
        return p
    end

    # convert primitive to conservative variables
    @inline function prim2cons(prim, equations::CompressibleEulerEquationsMs1T2D) 
        (v1, v2, T, rhos...) = prim

        rho = 0.0
        @inbounds for i in eachcomponent(equations.thermodata)
            rho += rhos[i]
        end
        rho_v1 = rho * v1
        rho_v2 = rho * v2
        rho_e = rho * energy_from_rho_vec(rhos, rho, T, equations.thermodata) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
        return SVector(rho_v1, rho_v2, rho_e, rhos...)
    end

    @inline function Trixi.cons2prim(u, equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e = u 

        @inbounds prim_rho = SVector{ncomponents(equations), Float64}(u[i + 3]
                                                                      for i in eachcomponent(equations))

        rho = density(u, equations)
        rho_inv = 1.0 / rho
        v1 = rho_v1 * rho_inv
        v2 = rho_v2 * rho_inv

        e_internal = (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2)) * rho_inv

        T = temperature(u, rho, rho_inv, e_internal, equations)
        prim_other = SVector{3, Float64}(v1, v2, T)
    
        return vcat(prim_other, prim_rho)
    end

    @inline function pressure(T, u, equations::CompressibleEulerEquationsMs1T2D)
        nrho = number_density(u, equations)
        
        p = T * nrho
        return p
    end

    # Convert conservative variables to primitive
    @inline function cons2prim_with_index(u, equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e = u 

        @inbounds prim_rho = SVector{ncomponents(equations.thermodata), Float64}(u[i + 3]
                                                                        for i in eachcomponent(equations.thermodata))

        rho = density(u, equations)
        rho_inv = 1.0 / rho
        v1 = rho_v1 * rho_inv
        v2 = rho_v2 * rho_inv

        e_internal = (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2)) * rho_inv

        T, index_lower_e, fracpos_e, index_lower_c, fracpos_c = temperature_rho_inv_with_index(u, rho_inv, 0.28*e_internal, e_internal, equations.thermodata)
        prim_other = SVector{3, Float64}(v1, v2, T)

        # both index pairs are returned: with CvOffset the energy and the c_v tables
        # are not tabulated at the same temperatures, so they need separate stencils
        return index_lower_e, fracpos_e, index_lower_c, fracpos_c, vcat(prim_other, prim_rho)
    end

    function varnames(::typeof(cons2cons),
                      equations::CompressibleEulerEquationsMs1T2D)
        cons = ("rho_v1", "rho_v2", "rho_e")
        rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations.thermodata)))
        return (cons..., rhos...)
    end

    @inline function entropy_thermodynamic(u,rho, equations::CompressibleEulerEquationsMs1T2D)
        #e = energy_internal_without_rho(cons, equations)  # get e
        T = temperature(u, equations::CompressibleEulerEquationsMs1T2D)
        s = entropy_c_v_integral(u, T, rho, equations) 
        @inbounds for i in eachcomponent(equations)
            s -= (u[i + 3] / rho) * log(u[i+3]) * equations.inv_mass[i] 
        end
        return s
    end

    @inline function entropy_thermodynamic(u, equations::CompressibleEulerEquationsMs1T2D)
        
        #e = energy_internal_without_rho(cons, equations)  # get e
        rho=density(u, equations)
        return entropy_thermodynamic(u,rho, equations)
    end

    @inline function entropy_math(u, equations::CompressibleEulerEquationsMs1T2D)
        rho=density(u, equations)
        return -rho * entropy_thermodynamic(u, rho, equations)
    end

    @inline function entropy(u, equations::CompressibleEulerEquationsMs1T2D )
        return entropy_math(u, equations)
    end

    @inline function cons2entropy(u, equations::CompressibleEulerEquationsMs1T2D)
            rho_v1, rho_v2, _ = u
            rho = density(u, equations)
            v1 = rho_v1 / rho
            v2 = rho_v2 / rho
            T = temperature(u, equations)
            T_inv = 1/T
            entr_other = SVector{3, real(equations)}(v1*T_inv, v2*T_inv, -T_inv)
            minus_v_half_by_T = -(v1^2 + v2^2)/2*T_inv

            index_lower_e, fracpos_e, index_lower_c, _ = get_index_lower_fracpos(T, equations.thermodata)
            @inbounds entr_rho = SVector{ncomponents(equations.thermodata), real(equations)}(-entropy_c_v_integral_component(i, index_lower_c, T, equations.thermodata) + log(abs(u[i + 3]))*equations.thermodata.inv_mass[i] + energy_component(i, index_lower_e, fracpos_e, equations.thermodata)/T + minus_v_half_by_T
                                                                        for i in eachcomponent(equations.thermodata))
            return vcat(entr_other, entr_rho)
        end

    @inline function Base.real(::CompressibleEulerEquationsMs1T2D)
        Float64
    end

    # entropy-conservative flux
    # see Oblapenko, Torrilhon, Computers and Fluids 2025 106640, DOI 10.1016/j.compfluid.2025.106640
    # and Oblapenko, Tarnovskiy, Ertl, Torrilhon, STAB/DGLR Symposium 2024, DOI 10.1007/978-3-032-11115-9_36
    @inline function flux_oblapenko(u_ll, u_rr, orientation::Integer,
            equations::CompressibleEulerEquationsMs1T2D)
        thermodata = equations.thermodata
        # `ie`/`fe` index the energy table, `ic`/`fc` the c_v table; the two coincide
        # only for NoCvOffset
        # the c_v fractional positions are not needed here: the entropy integral only
        # takes the cell index, and c_v itself is only evaluated at T_mid below
        (ie_ll, fe_ll, ic_ll, _, (v1_ll, v2_ll, T_ll, rhos_ll...)) = cons2prim_with_index(u_ll, equations)
        (ie_rr, fe_rr, ic_rr, _, (v1_rr, v2_rr, T_rr, rhos_rr...)) = cons2prim_with_index(u_rr, equations)

        v1_avg = 0.5*(v1_ll + v1_rr)
        v2_avg = 0.5*(v2_ll + v2_rr)
        inv_T_avg = 0.5 * (1.0 / T_ll + 1.0 / T_rr)
        T_geo_sqr = T_ll * T_rr

        velocity_square_avg = 0.5 * (v1_ll^2 + v2_ll^2 + v1_rr^2 + v2_rr^2)
        T_jump = T_rr - T_ll

        tmp_sum = 0.0
        @inbounds for i in eachcomponent(thermodata)
            tmp_sum = tmp_sum + 0.5*((abs(rhos_ll[i])+abs(rhos_rr[i]))*thermodata.inv_mass[i])
        end

        if(orientation == 1)
            @inbounds fx_rhos = SVector{ncomponents(thermodata), Float64}(ln_mean(abs(rhos_ll[i]), abs(rhos_rr[i])) * v1_avg
                                                                for i in eachcomponent(thermodata))  #use ln_mean function in math.jl
            fx_rhos_sum = sum(fx_rhos)                                      
            fx_rho_v1 = v1_avg * fx_rhos_sum  + tmp_sum / inv_T_avg
            fx_rho_v2 = v2_avg * fx_rhos_sum
            fx_e = v1_avg * fx_rho_v1 + v2_avg * fx_rho_v2 - 0.5 * fx_rhos_sum * velocity_square_avg

            if (abs(T_jump) >= equations.min_T_jump)
                inv_T_jump = 1.0 / T_jump
                @inbounds for i in eachcomponent(thermodata)
                    cv_Tast_over_Tast = (entropy_c_v_integral_component(i, ic_rr, T_rr, thermodata)
                                            - entropy_c_v_integral_component(i, ic_ll, T_ll, thermodata)) * inv_T_jump
                    e_int_ll = energy_component(i, ie_ll, fe_ll, thermodata)
                    e_int_rr = energy_component(i, ie_rr, fe_rr, thermodata)
                    cv_T_astast = (e_int_rr - e_int_ll) * inv_T_jump
                    
                    fx_e = fx_e + fx_rhos[i] * (0.5*(e_int_ll+e_int_rr) + T_geo_sqr * (cv_Tast_over_Tast - inv_T_avg* cv_T_astast))     
                end
            else
                T_mid = 0.5 * (T_ll + T_rr)
                inv_T_mid = 1.0 / T_mid

                _, _, ic_mid, fc_mid = get_index_lower_fracpos(T_mid, thermodata)
                @inbounds for i in eachcomponent(thermodata)
                    cvmid = c_v_component(i, ic_mid, fc_mid, thermodata)    
                    fx_e = fx_e + fx_rhos[i] * (0.5*(energy_component(i, ie_ll, fe_ll, thermodata) +
                                                energy_component(i, ie_rr, fe_rr, thermodata))
                                            + T_geo_sqr * (cvmid * inv_T_mid - inv_T_avg * cvmid))
                end
            end
        else
            @inbounds fx_rhos = SVector{ncomponents(thermodata), Float64}(ln_mean(abs(rhos_ll[i]), abs(rhos_rr[i])) * v2_avg
                                                                for i in eachcomponent(thermodata))
            fx_rhos_sum = sum(fx_rhos)                                      
            fx_rho_v2 = v2_avg * fx_rhos_sum  + tmp_sum / inv_T_avg
            fx_rho_v1 = v1_avg * fx_rhos_sum
            fx_e = v1_avg * fx_rho_v1 + v2_avg * fx_rho_v2 - 0.5 * fx_rhos_sum * velocity_square_avg
            if (abs(T_jump) >= equations.min_T_jump)
                inv_T_jump = 1.0  / T_jump
                @inbounds for i in eachcomponent(thermodata)
                    cv_Tast_over_Tast = (entropy_c_v_integral_component(i, ic_rr, T_rr, thermodata)
                                            - entropy_c_v_integral_component(i, ic_ll, T_ll, thermodata)) * inv_T_jump
                    e_int_ll = energy_component(i, ie_ll, fe_ll, thermodata)
                    e_int_rr = energy_component(i, ie_rr, fe_rr, thermodata)

                    cv_T_astast = (e_int_rr - e_int_ll) * inv_T_jump

                    fx_e += fx_rhos[i] * (0.5*(e_int_ll+e_int_rr) + T_geo_sqr * (cv_Tast_over_Tast - inv_T_avg* cv_T_astast))
                end
            else
                T_mid = 0.5 * (T_ll + T_rr)
                inv_T_mid = 1.0 / T_mid
                _, _, ic_mid, fc_mid = get_index_lower_fracpos(T_mid, thermodata)
                @inbounds for i in eachcomponent(thermodata)
                    cvmid = c_v_component(i, ic_mid, fc_mid, thermodata)        

                    fx_e += fx_rhos[i] * (0.5*(energy_component(i, ie_ll, fe_ll, thermodata) +
                                                energy_component(i, ie_rr, fe_rr, thermodata))
                                            + T_geo_sqr * (cvmid  * inv_T_mid - inv_T_avg * cvmid))                        
                end
            end
        end
        return SVector(fx_rho_v1, fx_rho_v2, fx_e, fx_rhos...)
    end

    @inline function flux_oblapenko(u_ll, u_rr, normal_direction::AbstractVector,
        equations::CompressibleEulerEquationsMs1T2D)

        # (v1_ll, v2_ll, T_ll, rhos_ll...) = cons2prim(u_ll, equations)
        # (v1_rr, v2_rr, T_rr, rhos_rr...) = cons2prim(u_rr, equations)
    
        (il_ll, fp_ll, (v1_ll, v2_ll, T_ll, rhos_ll...)) = cons2prim_with_index(u_ll, equations)
        (il_rr, fp_rr, (v1_rr, v2_rr, T_rr, rhos_rr...)) = cons2prim_with_index(u_rr, equations)

        v1_avg = 0.5*(v1_ll + v1_rr)
        v2_avg = 0.5*(v2_ll + v2_rr)
        inv_T_avg = 0.5 * (1.0 / T_ll + 1.0 / T_rr)
        T_geo_sqr = T_ll * T_rr

        velocity_square_avg = 0.5 * (v1_ll^2 + v2_ll^2 + v1_rr^2 + v2_rr^2)
        T_jump = T_rr - T_ll

        # il_ll, fp_ll = get_index_lower_fracpos(T_ll, equations)
        # il_rr, fp_rr = get_index_lower_fracpos(T_rr, equations)

        v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
        v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

        tmp_sum = 0.0
        @inbounds for i in eachcomponent(equations)
            tmp_sum = tmp_sum + 0.5*((abs(rhos_ll[i])+abs(rhos_rr[i]))*equations.inv_mass[i])
        end

        v_dot_n_avg = 0.5f0 * (v_dot_n_ll + v_dot_n_rr)

        @inbounds fx_rhos = SVector{ncomponents(equations), Float64}(Trixi.ln_mean(abs(rhos_ll[i]), abs(rhos_rr[i])) *
                                                           v_dot_n_avg for i in eachcomponent(equations))
                                                           #use ln_mean function in math.jl
        # fx_rhos_sum = sum(fx_rhos)
        fx_rhos_sum = 0.0
        @inbounds for i in eachcomponent(equations)
            fx_rhos_sum += fx_rhos[i]
        end
        
        p_avg = tmp_sum / inv_T_avg
        fx_rho_v1 = v1_avg * fx_rhos_sum + p_avg * normal_direction[1]
        fx_rho_v2 = v2_avg * fx_rhos_sum + p_avg * normal_direction[2]
        fx_e = v1_avg * fx_rho_v1 + v2_avg * fx_rho_v2 - 0.5 * fx_rhos_sum * velocity_square_avg

        if (abs(T_jump) >= equations.min_T_jump)
            inv_T_jump = 1.0 / T_jump
            @inbounds for i in eachcomponent(equations)
                cv_Tast_over_Tast = (entropy_c_v_integral(i, il_rr, fp_rr, T_rr, equations)
                                         - entropy_c_v_integral(i, il_ll, fp_ll, T_ll, equations)) * inv_T_jump

                e_int_ll = energy_component(i, il_ll, fp_ll, equations) 
                e_int_rr = energy_component(i, il_rr, fp_rr, equations) 
                cv_T_astast = (e_int_rr - e_int_ll) * inv_T_jump
                
                fx_e = fx_e + fx_rhos[i] * (0.5*(e_int_ll+e_int_rr) + T_geo_sqr * (cv_Tast_over_Tast - inv_T_avg* cv_T_astast))
            end
        else
            T_mid = 0.5 * (T_ll + T_rr)
            inv_T_mid = 1.0 / T_mid
            il_mid, fp_mid = get_index_lower_fracpos(T_mid, equations)
            @inbounds for i in eachcomponent(equations)
                cvmid = c_v(i, il_mid, fp_mid, equations)     
                # cvmid = c_v(i, T_mid, equations)        
                fx_e = fx_e + fx_rhos[i] * (0.5*(energy_component(i, il_ll, fp_ll, equations) + energy_component(i, il_rr, fp_rr,equations)) + T_geo_sqr * (cvmid  * (inv_T_mid - inv_T_avg)))
            end
        end
        return SVector(fx_rho_v1, fx_rho_v2, fx_e, fx_rhos...)
    end

    @inline function flux(u, orientation::Integer,
                      equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e = u

        rhoinv = 1.0 / density(u, equations)

        v1 = rho_v1 * rhoinv
        v2 = rho_v2 * rhoinv

        T = temperature_rhoinv(u, rhoinv, equations)
        p = pressure(T, u, equations)

        if orientation == 1
            f_rho = densities(u, v1, equations)
            f1 = rho_v1 * v1 + p
            f2 = rho_v2 * v1
            f3 = (rho_e + p) * v1
        else
            f_rho = densities(u, v2, equations)
            f1 = rho_v1 * v2
            f2 = rho_v2 * v2 + p
            f3 = (rho_e + p) * v2
        end

        f_other = SVector(f1, f2, f3)

        return vcat(f_other, f_rho)
    end

    # Calculate 1D flux for a single point
    @inline function flux(u, normal_direction::AbstractVector,
                        equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e = u

        rhoinv = 1.0 / density(u, equations)

        v1 = rho_v1 * rhoinv
        v2 = rho_v2 * rhoinv
        v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]

        T = temperature_rhoinv(u, rhoinv, equations)
        p = pressure(T, u, equations)

        f_rho = densities(u, v_normal, equations)
        f1 = rho_v1 * v_normal + p * normal_direction[1]
        f2 = rho_v2 * v_normal + p * normal_direction[2]
        f3 = (rho_e + p) * v_normal

        f_other = SVector(f1, f2, f3)

        return vcat(f_other, f_rho)
    end

    @inline function max_abs_speeds(u, equations::CompressibleEulerEquationsMs1T2D)
        # println(u)
        (index_lower, fracpos, (v1, v2, T, rhos...)) = cons2prim_with_index(u, equations)
        # rho = sum(rhos)

        rho = 0.0
        @inbounds for i in eachcomponent(equations)
            rho += rhos[i]
        end
        # c = sqrt(gamma * p / rho)
        gamma = get_gamma(u, rho, index_lower, fracpos, equations)
        p = pressure(T, u, equations)
        
        c = sqrt(gamma * p / rho)
        return abs(v1) + c, abs(v2) + c
    end

    @inline function max_abs_speed(u_ll, u_rr, orientation::Integer,
                                         equations::CompressibleEulerEquationsMs1T2D)
        _, v1_ll, v2_ll, _, T_ll = cons2prim(u_ll, equations)
        _, v1_rr, v2_rr, _, T_rr = cons2prim(u_rr, equations)
        gamma_ll = get_gamma(T_ll, equations)
        gamma_rr = get_gamma(T_rr, equations)
    
        # Get the velocity value in the appropriate direction
        if orientation == 1
            v_ll = v1_ll
            v_rr = v1_rr
        else # orientation == 2
            v_ll = v2_ll
            v_rr = v2_rr
        end
        # Calculate sound speeds
        # p = rho * T because everything is scaled
        # c_ll = sqrt(gamma_ll * p_ll / rho_ll)
        # c_rr = sqrt(gamma_rr * p_rr / rho_rr)
        c_ll = sqrt(gamma_ll * T_ll)
        c_rr = sqrt(gamma_rr * T_rr)
    
        λ_max = max(abs(v_ll) + c_ll, abs(v_rr) + c_rr)
        return λ_max
    end

    @inline function max_abs_speed(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::CompressibleEulerEquationsMs1T2D)


        (il_ll, fp_ll, (v1_ll, v2_ll, T_ll, rhos_ll...)) = cons2prim_with_index(u_ll, equations)
        (il_rr, fp_rr, (v1_rr, v2_rr, T_rr, rhos_rr...)) = cons2prim_with_index(u_rr, equations)

        # rho_ll = sum(rhos_ll)
        # rho_rr = sum(rhos_rr)

        rho_ll = 0.0
        rho_rr = 0.0
        for i in eachcomponent(equations)
            rho_ll += rhos_ll[i]
            rho_rr += rhos_rr[i]
        end

        gamma_ll = get_gamma(u_ll, rho_ll, il_ll, fp_ll, equations)
        gamma_rr = get_gamma(u_rr, rho_rr, il_rr, fp_rr, equations)
    
        p_ll = pressure(T_ll, u_ll, equations)
        p_rr = pressure(T_rr, u_rr, equations)
        # Calculate normal velocities and sound speed
        # left
        v_ll = (v1_ll * normal_direction[1]
                +
                v2_ll * normal_direction[2])
        # c_ll = sqrt(gamma_ll * p_ll / rho_ll)
        c_ll = sqrt(gamma_ll * p_ll / rho_ll)
        # right
        v_rr = (v1_rr * normal_direction[1]
                +
                v2_rr * normal_direction[2])
        # c_rr = sqrt(gamma_rr * p_rr / rho_rr)
        c_rr = sqrt(gamma_rr * p_rr / rho_rr)
    
        norm_norm = norm(normal_direction)
        return max(abs(v_ll) + c_ll * norm_norm, abs(v_rr) + c_rr * norm_norm)
    end

    @inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                                  x, t,
                                                  surface_flux_function,
                                                  equations::CompressibleEulerEquationsMs1T2D)
        norm_ = norm(normal_direction)
        # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
        # normal = normal_direction / norm_
    
        # rotate the internal solution state
        # u_local = Trixi.rotate_to_x(u_inner, normal, equations)
    
        # compute the primitive variables
        # rho_local, v_normal, v_tangent, p_local, T_local = cons2prim(u_local, equations)
        (v_x, v_y, T_local, rhos_local...) = cons2prim(u_inner, equations)

        # c = normal_vector[1]
        # s = normal_vector[2]
    
        # # Apply the 2D rotation matrix with normal and tangent directions of the form
        # # [ 1    0    0   0;
        # #   0   n_1  n_2  0;
        # #   0   t_1  t_2  0;
        # #   0    0    0   1 ]
        # # where t_1 = -n_2 and t_2 = n_1
    
        v_normal = (normal_direction[1] * v_x + normal_direction[2] * v_y) / norm_
        # rho_local = sum(rhos_local)
        rho_local = 0.0
        @inbounds for i in eachcomponent(equations)
            rho_local += rhos_local[i]
        end
        gamma_local = get_gamma(u_inner, rho_local, T_local, equations)
    
        p_local = pressure(T_local, u_inner, equations)
        # Get the solution of the pressure Riemann problem
        # See Section 6.3.3 of
        # Eleuterio F. Toro (2009)
        # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
        # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
        if v_normal <= 0.0
            sound_speed = sqrt(gamma_local * p_local / rho_local) # local sound speed
            p_star = p_local *
                     (1 + 0.5 * (gamma_local - 1) * v_normal / sound_speed)^(2 *
                                                                            gamma_local / (gamma_local - 1))
        else # v_normal > 0.0
            A = 2 / ((gamma_local + 1) * rho_local)
            B = p_local * (gamma_local - 1) / (gamma_local + 1)
            p_star = p_local +
                     0.5 * v_normal / A *
                     (v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
        end
    
        # For the slip wall we directly set the flux as the normal velocity is zero
        return SVector(p_star * normal_direction[1],
                       p_star * normal_direction[2],
                       0.0,
                       zeros(ncomponents(equations))...)
    end
    
    """
        boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                     surface_flux_function, equations::CompressibleEulerEquations2D)
    
    Should be used together with [`TreeMesh`](@ref).
    """
    @inline function boundary_condition_slip_wall(u_inner, orientation,
                                                  direction, x, t,
                                                  surface_flux_function,
                                                  equations::CompressibleEulerEquationsMs1T2D)

        # println("Sflux3")
        # get the appropriate normal vector from the orientation
        if orientation == 1
            normal_direction = SVector(1, 0)
        else # orientation == 2
            normal_direction = SVector(0, 1)
        end
    
        # compute and return the flux using `boundary_condition_slip_wall` routine above
        return boundary_condition_slip_wall(u_inner, normal_direction, direction,
                                            x, t, surface_flux_function, equations)
    end
    
    """
        boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
                                     surface_flux_function, equations::CompressibleEulerEquations2D)
    
    Should be used together with [`StructuredMesh`](@ref).
    """
    @inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                                  direction, x, t,
                                                  surface_flux_function,
                                                  equations::CompressibleEulerEquationsMs1T2D)
        # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
        # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
        if isodd(direction)
            boundary_flux = -boundary_condition_slip_wall(u_inner, -normal_direction,
                                                          x, t, surface_flux_function,
                                                          equations)
        else
            boundary_flux = boundary_condition_slip_wall(u_inner, normal_direction,
                                                         x, t, surface_flux_function,
                                                         equations)
        end
    
        return boundary_flux
    end
end # @muladd