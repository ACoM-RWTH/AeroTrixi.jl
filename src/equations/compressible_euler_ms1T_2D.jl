@muladd begin
    struct CompressibleEulerEquationsMs1T2D{I, CvO, NVARS, NCOMP} <: 
        Trixi.AbstractCompressibleEulerMulticomponentEquations{2, NVARS, NCOMP}
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
                    error("Offset of c_v(T) not implemented")
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
        rho = zero(u[1])

        @inbounds for i in eachcomponent(equations.thermodata)
            rho += u[i + 3]
        end

        return rho
    end

    # Calculate total energy for a conservative state `cons`
    @inline energy_total(cons, ::CompressibleEulerEquationsMs1T2D) = cons[3]  #so rhis is just rho_e?    

    # Calculate kinetic energy for a conservative state `cons`
    @inline function energy_kinetic(u, rho, equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, _ = u
        #rho = density(u, equations)
        return (rho_v1^2 + rho_v2^2) / (2 * rho)
    end

    @inline function energy_kinetic(u, equations::CompressibleEulerEquationsMs1T2D)
        #  rho_v1, rho_v2, rho_e = u
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

    @inline function temperature(u, equations::CompressibleEulerEquationsMs1T2D)
        rho = density(u, equations)
        eint = energy_internal(u, equations) / rho
        return temperature_rhoinv(u, 1.0/rho, 0.28*eint, eint, equations.thermodata)
    end

    @inline function density_and_number_density(u, equations::CompressibleEulerEquationsMs1T2D)
        rho = zero(u[1])
        nrho = zero(u[1])

        @inbounds for i in eachcomponent(equations.thermodata)
            rho += u[i + 3]
            nrho += u[i + 3] * equations.thermodata.inv_mass[i]
        end

        return rho, nrho
    end

    @inline function Trixi.pressure(u, equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e, _ = u
        rho, nrho = density_and_number_density(u, equations)
        invrho = 1.0 / rho
        e_internal = (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) * invrho) * invrho
        p = temperature_rhoinv(u, invrho, 0.28*e_internal, e_internal, equations.thermodata) * nrho
        return p
    end



    @inline function Trixi.prim2cons(prim, equations::CompressibleEulerEquationsMs1T2D) 
        (v1, v2, T, rhos...) = prim
        # rho=sum(rhos)

        rho = 0.0
        @inbounds for i in eachcomponent(equations.thermodata)
            rho += rhos[i]
        end
        rho_v1 = rho * v1
        rho_v2 = rho * v2
        # rho_e = rho * energy_from_rho_vec(rhos, rho, T, equations) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
        rho_e = rho * energy_from_rho_vec(rhos, rho, T, equations.thermodata) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
        return SVector(rho_v1, rho_v2, rho_e, rhos...)
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


        T, index_lower, fracpos = temperature_rhoinv_with_index(u, rho_inv, 0.28*e_internal, e_internal, equations.thermodata)
        prim_other = SVector{3, Float64}(v1, v2, T)

        return index_lower, fracpos, vcat(prim_other, prim_rho)
    end

    function Trixi.varnames(::typeof(cons2cons),
                        equations::CompressibleEulerEquationsMs1T2D)
        cons = ("rho_v1", "rho_v2", "rho_e")
        rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations.thermodata))) # TODO: pre-alloc?
        return (cons..., rhos...)
    end

    @inline function Trixi.cons2entropy(u, equations::CompressibleEulerEquationsMs1T2D)
            #energy(i, T, equations
            #ncomponents(equations)
            rho_v1, rho_v2, rho_e = u
            #prim_rho = SVector{ncomponents(equations), real(equations)}(u[i + 3]
            #                                                            for i in eachcomponent(equations))
            #rho, nrho = density(u, equations)
            rho = density(u, equations)
            v1 = rho_v1 / rho
            v2 = rho_v2 / rho
            T = temperature(u, equations)
            entr_other = SVector{3, real(equations)}(v1/T, v2/T, -1/T)
            minus_v_half_by_T = -(v1^2 + v2^2)/2/T
            # entr_rho = SVector{ncomponents(equations), real(equations)}(-entropy_c_v_integral(i, T,equations) + log(u[i + 3])/equations.mass[i] + energy_component(i, T, equations)/T + minus_v_half_by_T
            #                                                             for i in eachcomponent(equations))
            index_lower, fracpos = get_index_lower_fracpos(T, equations.thermodata)
            @inbounds entr_rho = SVector{ncomponents(equations.thermodata), real(equations)}(-entropy_c_v_integral_component(i, index_lower, fracpos, T, equations.thermodata) + log(abs(u[i + 3]))*equations.thermodata.inv_mass[i] + energy_component(i, index_lower, fracpos, equations.thermodata)/T + minus_v_half_by_T
                                                                        for i in eachcomponent(equations.thermodata))
            return vcat(entr_other, entr_rho)
        end

    @inline function Base.real(::CompressibleEulerEquationsMs1T2D{NVARS, NCOMP}) where {NVARS,
                                                                                            NCOMP}
        Float64
    end



    @inline function flux_oblapenko(u_ll, u_rr, orientation::Integer,
            equations::CompressibleEulerEquationsMs1T2D)

        # (v1_ll, v2_ll, T_ll, rhos_ll...) = cons2prim(u_ll, equations)
        # (v1_rr, v2_rr, T_rr, rhos_rr...) = cons2prim(u_rr, equations)
        thermodata = equations.thermodata
        (il_ll, fp_ll, (v1_ll, v2_ll, T_ll, rhos_ll...)) = cons2prim_with_index(u_ll, equations)
        (il_rr, fp_rr, (v1_rr, v2_rr, T_rr, rhos_rr...)) = cons2prim_with_index(u_rr, equations)
        #rho = density()

        # rhos_ll = abs.(rhos_ll)
        # rhos_rr = abs.(rhos_rr)

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

        # il_ll, fp_ll = get_index_lower_fracpos(T_ll, equations)
        # il_rr, fp_rr = get_index_lower_fracpos(T_rr, equations)
        # energy_component(i, index_lower, fracpos, equations)
        # entropy_c_v_integral(i::Int64, index_lower, fracpos, T_b, equations::CompressibleEulerEquationsMs1T2D)
        # function c_v(i, index_lower, fracpos, equations::CompressibleEulerEquationsMs1T2D)

        if(orientation == 1)
            @inbounds fx_rhos = SVector{ncomponents(thermodata), Float64}(Trixi.ln_mean(abs(rhos_ll[i]), abs(rhos_rr[i])) * v1_avg
                                                                for i in eachcomponent(thermodata))  #use ln_mean function in math.jl
            fx_rhos_sum = sum(fx_rhos)                                      
            fx_rho_v1 = v1_avg * fx_rhos_sum  + tmp_sum / inv_T_avg
            fx_rho_v2 = v2_avg * fx_rhos_sum
            fx_e = v1_avg * fx_rho_v1 + v2_avg * fx_rho_v2 - 0.5 * fx_rhos_sum * velocity_square_avg

            if (abs(T_jump) >= equations.min_T_jump)
                inv_T_jump = 1.0 / T_jump
                @inbounds for i in eachcomponent(thermodata)
                    cv_Tast_over_Tast = (entropy_c_v_integral_component(i, il_rr, fp_rr, T_rr, thermodata)
                                            - entropy_c_v_integral_component(i, il_ll, fp_ll, T_ll, thermodata)) * inv_T_jump
                    # cv_Tast_over_Tast = (entropy_c_v_integral(i, T_rr, equations)
                    #                      - entropy_c_v_integral(i, T_ll, equations)) * inv_T_jump

                    # e_int_ll = energy_component(i, T_ll, equations) 
                    # e_int_rr = energy_component(i, T_rr, equations) 
                    e_int_ll = energy_component(i, il_ll, fp_ll, thermodata)
                    e_int_rr = energy_component(i, il_rr, fp_rr, thermodata)
                    cv_T_astast = (e_int_rr - e_int_ll) * inv_T_jump
                    
                    fx_e = fx_e + fx_rhos[i] * (0.5*(e_int_ll+e_int_rr) + T_geo_sqr * (cv_Tast_over_Tast - inv_T_avg* cv_T_astast))     
                    # println("Ms,", entropy_c_v_integral(i, T_ll, equations), ", ", entropy_c_v_integral(i, T_rr, equations), ", ", T_ll, ", ", T_rr, ", ", 
                    # cv_Tast_over_Tast, ", ", inv_T_avg * cv_T_astast, ", ", cv_T_astast,
                    # ", ", (e_int_rr - e_int_ll), ", ", (e_int_rr - e_int_ll) * inv_T_jump)
                end
            else
                T_mid = 0.5 * (T_ll + T_rr)
                inv_T_mid = 1.0 / T_mid

                il_mid, fp_mid = get_index_lower_fracpos(T_mid, thermodata)
                @inbounds for i in eachcomponent(thermodata)
                    # cvmid = c_v(i, T_mid, equations)        
                    cvmid = c_v_component(i, il_mid, fp_mid, thermodata)    
                    # fx_e += fx_rhos[i] * (0.5*(energy_component(i, T_ll, equations) +
                    #                            energy_component(i, T_rr, equations))
                    #                       + T_geo_sqr * (cvmid  * inv_T_mid - inv_T_avg * cvmid))
                    # Add before the problematic line
                    fx_e = fx_e + fx_rhos[i] * (0.5*(energy_component(i, il_ll, fp_ll, thermodata) +
                                                energy_component(i, il_rr, fp_rr, thermodata))
                                            + T_geo_sqr * (cvmid * inv_T_mid - inv_T_avg * cvmid))
                end
            end
        else
            @inbounds fx_rhos = SVector{ncomponents(thermodata), Float64}(Trixi.ln_mean(abs(rhos_ll[i]), abs(rhos_rr[i])) * v2_avg
                                                                for i in eachcomponent(thermodata))
            fx_rhos_sum = sum(fx_rhos)                                      
            fx_rho_v2 = v2_avg * fx_rhos_sum  + tmp_sum / inv_T_avg
            fx_rho_v1 = v1_avg * fx_rhos_sum
            fx_e = v1_avg * fx_rho_v1 + v2_avg * fx_rho_v2 - 0.5 * fx_rhos_sum * velocity_square_avg
            if (abs(T_jump) >= equations.min_T_jump)
                inv_T_jump = 1.0  / T_jump
                @inbounds for i in eachcomponent(thermodata)
                    # cv_Tast_over_Tast = (entropy_c_v_integral(i, T_rr, equations) - entropy_c_v_integral(i, T_ll, equations)) * inv_T_jump
                    cv_Tast_over_Tast = (entropy_c_v_integral_component(i, il_rr, fp_rr, T_rr, thermodata)
                                            - entropy_c_v_integral_component(i, il_ll, fp_ll, T_ll, thermodata)) * inv_T_jump

                    # e_int_ll = energy_component(i, T_ll, equations) 
                    # e_int_rr = energy_component(i, T_rr, equations) 
                    e_int_ll = energy_component(i, il_ll, fp_ll, thermodata)
                    e_int_rr = energy_component(i, il_rr, fp_rr, thermodata)

                    cv_T_astast = (e_int_rr - e_int_ll) * inv_T_jump

                    fx_e += fx_rhos[i] * (0.5*(e_int_ll+e_int_rr) + T_geo_sqr * (cv_Tast_over_Tast - inv_T_avg* cv_T_astast))
                end
            else
                T_mid = 0.5 * (T_ll + T_rr)
                inv_T_mid = 1.0 / T_mid
                il_mid, fp_mid = get_index_lower_fracpos(T_mid, thermodata)
                @inbounds for i in eachcomponent(thermodata)
                    # cvmid = c_v(i, T_mid, equations)
                    cvmid = c_v_component(i, il_mid, fp_mid, thermodata)        

                    fx_e += fx_rhos[i] * (0.5*(energy_component(i, il_ll, fp_ll, thermodata) +
                                                energy_component(i, il_rr, fp_rr, thermodata))
                                            + T_geo_sqr * (cvmid  * inv_T_mid - inv_T_avg * cvmid))                        
                end
            end
        end
        return SVector(fx_rho_v1, fx_rho_v2, fx_e, fx_rhos...)
    end
end # @muladd