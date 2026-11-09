module TestGrad13Equations2D

using Test
using Trixi
using AeroTrixi
using Trixi: SVector

# Reference values computed with the finite-volume solver `MultiBlockMomentsFV`
# (src/Flux.c: Primitive, Conservative, Grad13Flux1D, Grad13Flux2D, Production, xFlux/yFlux
# with vanishing slopes, and the wall sign table WallSign of src/Scenario.c) for the two
# primitive states below, with Kn = 0.01.
const primA = SVector(1.3, 0.7, -0.4, 1.1, 1.3, 0.15, 0.95, 0.2, -0.3)
const primB = SVector(0.8, -0.2, 0.5, 0.9, 0.8, -0.1, 1.05, -0.15, 0.25)
const consA = SVector(1.3, 0.91, -0.52, 2.5675, 2.327, -0.169, 1.443, 3.10225, -1.6845)
const consB = SVector(0.8, -0.16, 0.4, 1.196, 0.672, -0.16, 1.04, -0.5572, 1.284)
const fluxA_x = SVector(0.91, 2.327, -0.169, 3.10225, 4.2349, -0.7778, 0.9341, 8.783825,
                        -1.003025)
const fluxA_y = SVector(-0.52, -0.169, 1.443, -1.6845, -0.7778, 0.9341, -1.9252,
                        -1.003025, 4.719925)
const fluxA_n = SVector(0.13, 1.261, 1.053, 0.51375, 1.9187, 0.2806, -0.9797, 4.467875,
                        3.174125) # direction (0.6, 0.8)
const prodA = SVector(0.0, 0.0, 0.0, 0.0,
                      -35.449739068151082, -26.587304301113345, 26.587304301113388,
                      -32.359248995543041, -1.9770046788007356)
const hllAB_x = SVector(0.93581992645423073, 2.6816523559778402, -1.1408307167461502,
                        2.8313984447310152, 3.7243983758939896, -0.19977819384997642,
                        0.75697256150132641, 9.3356026472401581, -4.023407097798982)
const hllAB_y = SVector(0.7087073199314915, 1.4548899700541278, -0.15604462547865408,
                        1.9141361529954897, 2.3659763447697899, 0.26265386357588349,
                        0.52418723419381252, 4.6753544555990034, -0.51799679560317791)
const mirrorA_x = SVector(1.3, -0.91, -0.52, 2.5675, 2.327, 0.169, 1.443, -3.10225,
                          -1.6845)

@testset "Grad13Equations2D" begin
    equations = Grad13Equations2D(0.01)
    @test Trixi.nvariables(equations) == 9
    @test length(Trixi.varnames(cons2cons, equations)) == 9
    @test length(Trixi.varnames(cons2prim, equations)) == 9
    @test length(Trixi.varnames(cons2moments, equations)) == 10

    @testset "variable conversions" begin
        @test prim2cons(primA, equations)≈consA rtol=1e-13
        @test prim2cons(primB, equations)≈consB rtol=1e-13
        @test cons2prim(consA, equations)≈primA rtol=1e-12
        @test cons2prim(prim2cons(primB, equations), equations)≈primB rtol=1e-12

        moments = cons2moments(consA, equations)
        @test moments[1:4] ≈ primA[1:4]
        @test moments[5] ≈ primA[1] * primA[4] # p = ρ θ
        @test moments[6:8] ≈ primA[1] .* primA[5:7] # P_ij = ρ T_ij
        @test moments[9:10] ≈ primA[8:9]

        @test Trixi.density(consA, equations) ≈ 1.3
        @test Trixi.temperature(consA, equations) ≈ 1.1
        @test Trixi.pressure(consA, equations) ≈ 1.3 * 1.1
        @test Trixi.density_pressure(consA, equations) ≈ 1.3^2 * 1.1
        @test Trixi.energy_total(consA, equations) ≈ 2.5675
        @test Trixi.energy_kinetic(consA, equations) +
              Trixi.energy_internal(consA, equations) ≈ 2.5675
    end

    @testset "fluxes" begin
        @test flux(consA, 1, equations)≈fluxA_x rtol=1e-12
        @test flux(consA, 2, equations)≈fluxA_y rtol=1e-12
        @test flux(consA, SVector(1.0, 0.0), equations)≈fluxA_x rtol=1e-12
        @test flux(consA, SVector(0.0, 1.0), equations)≈fluxA_y rtol=1e-12
        n = SVector(0.6, 0.8)
        @test flux(consA, n, equations)≈fluxA_n rtol=1e-12
        # the flux is linear in the (non-normalized) normal direction
        @test flux(consA, 2.5 * n, equations)≈2.5 * fluxA_n rtol=1e-12
        @test flux(consA, n,
                   equations)≈
        0.6 * flux(consA, 1, equations) + 0.8 * flux(consA, 2, equations) rtol=1e-12
        @test flux_central(consA, consB, n, equations) ≈
              0.5 * (flux(consA, n, equations) + flux(consB, n, equations))
    end

    @testset "rotations" begin
        n = SVector(0.6, 0.8)
        @test Trixi.rotate_to_x(consA, SVector(1.0, 0.0), equations) == consA
        @test Trixi.rotate_from_x(Trixi.rotate_to_x(consA, n, equations), n,
                                  equations)≈consA rtol=1e-13
        @test Trixi.rotate_to_x(Trixi.rotate_from_x(consB, n, equations), n,
                                equations)≈consB rtol=1e-13
        # rotation invariance: the rotated state has the normal velocity v⋅n and the
        # normal-normal component n⋅T n of the pressure tensor
        rho, v1, v2, theta, T11, T12, T22, q1, q2 = primA
        prim_rot = cons2prim(Trixi.rotate_to_x(consA, n, equations), equations)
        @test prim_rot[2] ≈ v1 * n[1] + v2 * n[2]
        @test prim_rot[5] ≈ T11 * n[1]^2 + 2 * T12 * n[1] * n[2] + T22 * n[2]^2
        @test prim_rot[5] + prim_rot[7] ≈ T11 + T22 # trace is invariant
        @test prim_rot[8] ≈ q1 * n[1] + q2 * n[2]
    end

    @testset "wave speeds and HLL flux" begin
        c(theta) = 2 * sqrt(5 / 3 * theta)
        @test all(Trixi.max_abs_speeds(consA, equations) .≈
                  (0.7 + c(1.1), 0.4 + c(1.1)))
        @test Trixi.min_max_speed_naive(consA, consB, 1, equations)[1] ≈ 0.7 - c(1.1)
        @test Trixi.min_max_speed_naive(consA, consB, 1, equations)[2] ≈ -0.2 + c(0.9)
        @test all(Trixi.min_max_speed_naive(consA, consB, SVector(2.0, 0.0), equations) .≈
                  2 .* Trixi.min_max_speed_naive(consA, consB, 1, equations))
        @test Trixi.max_abs_speed_naive(consA, consB, 2, equations) ≈ 0.5 + c(1.1)
        @test Trixi.max_abs_speed(consA, consB, 2, equations) ≈
              max(0.4 + c(1.1), 0.5 + c(0.9))

        flux_hll = FluxHLL(min_max_speed_naive)
        @test flux_hll(consA, consB, 1, equations)≈hllAB_x rtol=1e-12
        @test flux_hll(consA, consB, 2, equations)≈hllAB_y rtol=1e-12
        @test flux_hll(consA, consB, SVector(1.0, 0.0), equations)≈hllAB_x rtol=1e-12
        @test flux_hll(consA, consB, SVector(0.0, 1.0), equations)≈hllAB_y rtol=1e-12
        # consistency
        @test flux_hll(consA, consA, 1, equations)≈fluxA_x rtol=1e-12
        @test flux_lax_friedrichs(consA, consA, SVector(0.6, 0.8), equations)≈fluxA_n rtol=1e-12
        # the clipped HLL flux agrees away from the stationary sonic state
        @test flux_hll_grad13(consA, consB, 1, equations)≈hllAB_x rtol=1e-12
        @test flux_hll_grad13(consA, consB, 2, equations)≈hllAB_y rtol=1e-12
        @test flux_hll_grad13(consA, consB, SVector(0.0, 1.0), equations)≈hllAB_y rtol=1e-12
        @test flux_hll_grad13(consB, consA, SVector(-1.0, 0.0), equations)≈-hllAB_x rtol=1e-12
        # at the stationary sonic state (normal speed equal to the wave-speed bound), the wall
        # flux of the mirrored Riemann problem carries no mass; `FluxHLL` lets the gas out
        u_sonic = prim2cons(SVector(1.0, 2 * sqrt(5 / 3), 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
                                    0.0),
                            equations)
        u_mirror = specular_reflection(u_sonic, SVector(1.0, 0.0), equations)
        @test all(Trixi.min_max_speed_naive(u_sonic, u_mirror, 1, equations) .== 0)
        f_wall = flux_hll_grad13(u_sonic, u_mirror, 1, equations)
        @test abs(f_wall[1]) < 1e-14
        @test f_wall[2] ≈ 1.0 + (2 * sqrt(5 / 3))^2 # momentum flux of the interior state
        @test flux_hll(u_sonic, u_mirror, 1, equations)[1] ≈ 2 * sqrt(5 / 3)
    end

    @testset "source terms" begin
        x = SVector(0.0, 0.0)
        @test source_terms_grad13(consA, x, 0.0, equations)≈prodA rtol=1e-12
        @test collision_frequency(consA, equations) ≈ 1.3 * sqrt(1.1) / 0.01
        # no productions in equilibrium
        u_eq = prim2cons(SVector(2.0, 0.3, -0.1, 1.5, 1.5, 0.0, 1.5, 0.0, 0.0), equations)
        @test all(iszero, source_terms_grad13(u_eq, x, 0.0, equations))
        @test iszero(von_mises_stress(u_eq, equations))
        @test iszero(nonequilibrium(u_eq, equations))

        rho, v1, v2, theta, T11, T12, T22, q1, q2 = primA
        sigma = rho .* (T11 - theta, T12, T22 - theta)
        sigma_zz = -(sigma[1] + sigma[3])
        vm = sqrt(1.5 * (sigma[1]^2 + sigma[3]^2 + sigma_zz^2 + 2 * sigma[2]^2))
        @test von_mises_stress(consA, equations) ≈ vm
        p = rho * theta
        @test nonequilibrium(consA, equations) ≈
              sqrt((vm / p)^2 + (q1^2 + q2^2) / (p^2 * theta))
    end

    @testset "specular wall" begin
        @test specular_reflection(consA, SVector(1.0, 0.0), equations)≈mirrorA_x rtol=1e-13
        n = SVector(0.6, 0.8)
        u_mirror = specular_reflection(consA, n, equations)
        # mirroring twice is the identity
        @test specular_reflection(u_mirror, n, equations)≈consA rtol=1e-13
        prim = cons2prim(consA, equations)
        prim_mirror = cons2prim(u_mirror, equations)
        # density, temperature and the invariants are kept, the normal velocity and the
        # normal heat flux flip, the tangential ones are kept
        @test prim_mirror[1] ≈ prim[1]
        @test prim_mirror[4] ≈ prim[4]
        @test prim_mirror[2] * n[1] + prim_mirror[3] * n[2] ≈
              -(prim[2] * n[1] + prim[3] * n[2])
        @test -prim_mirror[2] * n[2] + prim_mirror[3] * n[1] ≈
              -prim[2] * n[2] + prim[3] * n[1]
        @test prim_mirror[8] * n[1] + prim_mirror[9] * n[2] ≈
              -(prim[8] * n[1] + prim[9] * n[2])
        @test prim_mirror[5] + prim_mirror[7] ≈ prim[5] + prim[7]

        # The wall flux of the HLL flux carries no mass and no normal heat flux
        flux_hll = FluxHLL(min_max_speed_naive)
        x = SVector(0.0, 0.0)
        f = boundary_condition_slip_wall(consA, n, x, 0.0, flux_hll, equations)
        @test abs(f[1]) < 1e-14
        @test abs(f[4]) < 1e-14
        # in coordinate directions, the TreeMesh version agrees with the P4estMesh version
        for orientation in 1:2, direction in (2 * orientation - 1, 2 * orientation)
            normal = orientation == 1 ? SVector(1.0, 0.0) : SVector(0.0, 1.0)
            sign = isodd(direction) ? -1 : 1
            f_tree = boundary_condition_slip_wall(consA, orientation, direction, x, 0.0,
                                                  flux_hll, equations)
            f_p4est = sign * boundary_condition_slip_wall(consA, sign * normal, x, 0.0,
                                                   flux_hll, equations)
            @test f_tree≈f_p4est rtol=1e-12
        end
    end
end

@testset "p4est_mesh_forward_facing_step" begin
    mesh = p4est_mesh_forward_facing_step(polydeg = 2)
    @test Trixi.ntrees(mesh) == 21
    @test sort(unique(mesh.boundary_names)) ==
          sort([Symbol("---"), :inflow, :outflow, :wall])
    @test count(==(:inflow), mesh.boundary_names) == 3
    @test count(==(:outflow), mesh.boundary_names) == 2
    @test count(==(:wall), mesh.boundary_names) == 3 + 9 + 1 + 6 # bottom, top, step face, step top
    @test_throws AssertionError p4est_mesh_forward_facing_step(polydeg = 2,
                                                               tree_size = 0.4)
end

end # module
