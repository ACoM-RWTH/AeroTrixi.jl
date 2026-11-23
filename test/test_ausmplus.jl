module TestAUSMPlus

using Test
using Trixi
using Trixi: CompressibleEulerEquations1D, flux, prim2cons
using AeroTrixi
using AeroTrixi: flux_ausmplus, SVector

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

end # module
