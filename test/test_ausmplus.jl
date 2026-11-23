module TestAUSMPlus

using Test
using Trixi
using Trixi: CompressibleEulerEquations1D, flux, prim2cons
using AeroTrixi
using AeroTrixi: flux_ausmplus, FluxAUSMPlusUp, flux_ausmplusup, SVector,
                 a_interface_ausm, a_interface_ausm_signed, a_star

@testset "AUSM+ flux: CompressibleEulerEquations1D" begin
    equations = CompressibleEulerEquations1D(1.4)
    orientation = 1

    @testset "consistency" begin
        # Subsonic, sonic, and supersonic states in both directions
        states = [SVector(1.1, 0.2, 2.5),
            SVector(0.7, -0.4, 1.8),
            SVector(1.0, 1.183216, 1.7), # |M| ≈ 1
            SVector(1.0, 3.0, 5.0),
            SVector(1.0, -3.0, 5.0)]

        for u in states
            @test flux_ausmplus(u, u, orientation, equations) ≈
                  flux(u, orientation, equations)
        end
    end

    # Different subsonic left and right states
    u_ll = prim2cons(SVector(1.2, 0.3, 1.5), equations)
    u_rr = prim2cons(SVector(0.8, -0.1, 0.9), equations)
    mirror(u) = SVector(u[1], -u[2], u[3])

    @testset "invariance under reflection" begin
        # Swapping and mirroring the states (v -> -v) flips the sign of the mass and
        # energy fluxes and keeps the momentum flux
        flip = SVector(-1, 1, -1)
        @test flux_ausmplus(mirror(u_rr), mirror(u_ll), orientation, equations) ≈
              flip .* flux_ausmplus(u_ll, u_rr, orientation, equations)
    end

    @testset "upwinding of supersonic flow" begin
        # Supersonic flow from the left/right: the flux is the physical flux of the
        # upwind state
        u_supersonic_ll = prim2cons(SVector(1.0, 3.0, 1.0), equations)
        u_supersonic_rr = prim2cons(SVector(0.5, 2.5, 0.7), equations)
        @test flux_ausmplus(u_supersonic_ll, u_supersonic_rr, orientation, equations) ≈
              flux(u_supersonic_ll, orientation, equations)
        @test flux_ausmplus(mirror(u_supersonic_rr), mirror(u_supersonic_ll),
                            orientation, equations) ≈
              flux(mirror(u_supersonic_ll), orientation, equations)
    end

    @testset "type stability" begin
        for RealT in (Float32, Float64)
            equations_RealT = CompressibleEulerEquations1D(RealT(1.4))
            # Use different subsonic states to exercise all terms of the flux
            u_ausm_ll = SVector(one(RealT), RealT(0.1), RealT(2))
            u_ausm_rr = SVector(RealT(0.5), RealT(-0.1), one(RealT))
            @test eltype(@inferred flux_ausmplus(u_ausm_ll, u_ausm_rr, orientation,
                                                 equations_RealT)) == RealT
        end
    end
end

@testset "AUSM+-up flux: CompressibleEulerEquations1D" begin
    equations = CompressibleEulerEquations1D(1.4)
    orientation = 1

    @testset "consistency" begin
        # Subsonic, sonic, and supersonic states in both directions
        states = [SVector(1.1, 0.2, 2.5),
            SVector(0.7, -0.4, 1.8),
            SVector(1.0, 1.183216, 1.7), # |M| ≈ 1
            SVector(1.0, 3.0, 5.0),
            SVector(1.0, -3.0, 5.0)]

        fluxes = [flux_ausmplusup,
            FluxAUSMPlusUp(mach_inf = 0.1),
            FluxAUSMPlusUp(mach_inf = 0.01, K_p = 0.5, K_u = 1.0, sigma = 0.5)]

        for u in states, flux_up in fluxes
            @test flux_up(u, u, orientation, equations) ≈ flux(u, orientation, equations)
        end

        @test_throws ArgumentError FluxAUSMPlusUp(mach_inf = 0.0)
    end

    # Different subsonic left and right states
    u_ll = prim2cons(SVector(1.2, 0.3, 1.5), equations)
    u_rr = prim2cons(SVector(0.8, -0.1, 0.9), equations)
    mirror(u) = SVector(u[1], -u[2], u[3])

    @testset "reduction to AUSM+" begin
        # Without the diffusion terms and without low-Mach scaling (f_a = 1, α = 3/16),
        # AUSM+-up reduces to AUSM+. This requires the same interface speed of sound,
        # which holds here since |v| < a* for both states, such that ã = a* in both
        # variants.
        flux_up_no_diffusion = FluxAUSMPlusUp(mach_inf = 1.0, K_p = 0.0, K_u = 0.0)
        @test flux_up_no_diffusion(u_ll, u_rr, orientation, equations) ≈
              flux_ausmplus(u_ll, u_rr, orientation, equations)
    end

    @testset "interface speed of sound" begin
        # AUSM+ uses |u_L| and |u_R|, AUSM+-up uses u_L and -u_R. Both agree if the flow
        # is directed towards the interface from both sides. In an expansion, only the
        # absolute values reduce a_{1/2}.
        a_abs(v1_ll, v1_rr) = a_interface_ausm(1.0, v1_ll, 1.0, 0.5, v1_rr, 0.7,
                                               equations)
        a_signed(v1_ll, v1_rr) = a_interface_ausm_signed(1.0, v1_ll, 1.0, 0.5, v1_rr,
                                                         0.7, equations)
        @test a_signed(3.0, -2.5) ≈ a_abs(3.0, -2.5)
        @test a_signed(-3.0, 2.5) > a_abs(-3.0, 2.5)
        # Signed variant in an expansion: no reduction, i.e., a_{1/2} = min(a*_L, a*_R)
        @test a_signed(-3.0, 2.5) ≈ min(a_star(1.0, -3.0, 1.0, equations),
                  a_star(0.5, 2.5, 0.7, equations))
    end

    @testset "invariance under reflection" begin
        # Swapping and mirroring the states (v -> -v) flips the sign of the mass and
        # energy fluxes and keeps the momentum flux
        flip = SVector(-1, 1, -1)
        for flux_up in (flux_ausmplusup, FluxAUSMPlusUp(mach_inf = 0.1))
            @test flux_up(mirror(u_rr), mirror(u_ll), orientation, equations) ≈
                  flip .* flux_up(u_ll, u_rr, orientation, equations)
        end
    end

    @testset "velocity diffusion" begin
        # p_u increases the pressure (momentum flux) under compression (v_ll > v_rr) and
        # decreases it under expansion. Density and pressure are equal such that only
        # p_u differs.
        u_compression_ll = prim2cons(SVector(1.0, 0.2, 1.0), equations)
        u_compression_rr = prim2cons(SVector(1.0, -0.2, 1.0), equations)
        flux_up_no_K_u = FluxAUSMPlusUp(K_u = 0.0)
        @test flux_ausmplusup(u_compression_ll, u_compression_rr, orientation,
                              equations)[2] >
              flux_up_no_K_u(u_compression_ll, u_compression_rr, orientation, equations)[2]
        @test flux_ausmplusup(u_compression_rr, u_compression_ll, orientation,
                              equations)[2] <
              flux_up_no_K_u(u_compression_rr, u_compression_ll, orientation, equations)[2]
    end

    @testset "pressure diffusion" begin
        # A pressure drop from left to right increases the mass flux to the right.
        # Density and velocity are equal such that only M_p differs.
        u_pressure_ll = prim2cons(SVector(1.0, 0.1, 1.2), equations)
        u_pressure_rr = prim2cons(SVector(1.0, 0.1, 0.8), equations)
        flux_up_no_K_p = FluxAUSMPlusUp(K_p = 0.0)
        @test flux_ausmplusup(u_pressure_ll, u_pressure_rr, orientation, equations)[1] >
              flux_up_no_K_p(u_pressure_ll, u_pressure_rr, orientation, equations)[1]

        # Smaller reference Mach numbers lead to more pressure diffusion (K_p / f_a)
        @test FluxAUSMPlusUp(mach_inf = 0.1)(u_pressure_ll, u_pressure_rr, orientation,
                                             equations)[1] >
              flux_ausmplusup(u_pressure_ll, u_pressure_rr, orientation, equations)[1]
    end

    @testset "upwinding of supersonic flow" begin
        # Supersonic flow from the left: the flux is the physical flux of u_ll
        u_supersonic_ll = prim2cons(SVector(1.0, 3.0, 1.0), equations)
        u_supersonic_rr = prim2cons(SVector(0.5, 2.5, 0.7), equations)
        @test flux_ausmplusup(u_supersonic_ll, u_supersonic_rr, orientation,
                              equations) ≈
              flux(u_supersonic_ll, orientation, equations)
    end

    @testset "type stability" begin
        for RealT in (Float32, Float64)
            equations_RealT = CompressibleEulerEquations1D(RealT(1.4))
            # Use different subsonic states to exercise all terms of the flux
            u_ausm_ll = SVector(one(RealT), RealT(0.1), RealT(2))
            u_ausm_rr = SVector(RealT(0.5), RealT(-0.1), one(RealT))
            @test eltype(@inferred flux_ausmplusup(u_ausm_ll, u_ausm_rr, orientation,
                                                   equations_RealT)) == RealT
            flux_up_low_mach = FluxAUSMPlusUp(mach_inf = RealT(0.1))
            @test eltype(@inferred flux_up_low_mach(u_ausm_ll, u_ausm_rr, orientation,
                                                    equations_RealT)) == RealT
        end
    end
end

end # module
