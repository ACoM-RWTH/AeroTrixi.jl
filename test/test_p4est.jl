module TestExamplesP4estMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "P4estMesh" begin
    @trixi_testset "elixir_euler_NACA0012airfoil_mach08.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_NACA0012airfoil_mach08.jl"),
                            l2=[
                                1.4825298121291054e-5,
                                1.1299955298727266e-5,
                                1.5024395928115724e-5,
                                3.9872656618032726e-5
                            ],
                            linf=[
                                0.6355977259450771,
                                1.1180698298562255,
                                0.6836203300405327,
                                1.4202190591669863
                            ],
                            tspan=(0.0, 0.1))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end

    @trixi_testset "elixir_navierstokes_NACA0012airfoil_mach08.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_navierstokes_NACA0012airfoil_mach08.jl"),
                            l2=[
                                0.0013329924272079746,
                                0.004524576282253344,
                                0.0024055822778919415,
                                0.01462132950009296
                            ],
                            linf=[
                                0.694673794075408,
                                1.7937600476958173,
                                1.0952815158719686,
                                6.725234662117499
                            ],
                            tspan=(0.0, 0.02))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end

    @trixi_testset "elixir_euler_poisson_blast_five_trees_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_poisson_blast_five_trees_amr.jl"),
                            l2=[
                                0.08348665697079459,
                                0.09963347982540754,
                                0.0996334798254072,
                                0.6313579377758771,
                                0.000578359869988509,
                                0.003042557094800779,
                                0.0030425570948007896
                            ],
                            linf=[
                                0.4853553503970873,
                                0.7365286060586608,
                                0.736528606058661,
                                2.761585821363732,
                                0.0023065950198078696,
                                0.0231217851821768,
                                0.023121785182176762
                            ],
                            tspan=(0.0, 0.05))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end

    @trixi_testset "elixir_grad13_forward_facing_step_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_grad13_forward_facing_step_amr.jl"),
                            l2=[
                                0.11023202877243597,
                                0.14396097049222742,
                                0.03588407793940817,
                                0.6536468221945436,
                                0.18002671333933623,
                                0.04616398051029551,
                                0.5729248352907773,
                                0.8238621977296843,
                                0.2903632271983952
                            ],
                            linf=[
                                1.9170284310504555,
                                2.5850613512673672,
                                2.570516720103975,
                                11.61649920996311,
                                4.486721555986335,
                                3.1594888800919,
                                9.966242651623423,
                                15.099478187611517,
                                22.515223654020485
                            ],
                            tspan=(0.0, 0.02))

        # 21 square trees; the elements around the face and the corner of the step are refined
        # to the maximum level from the start, the rest of the mesh follows the shocks
        @test Trixi.ntrees(mesh) == 21
        @test Trixi.nelements(semi.solver, semi.cache) == 2382
        # The solution stays in the physical range: positive density and pressure, and
        # the reflected shock does not exceed the Rankine-Hugoniot density ratio (3 for the
        # relative Mach number 3) by more than the overshoot of the moment shock structure
        let u = Trixi.wrap_array(sol.u[end], semi)
            rho = Array(u[1, :, :, :])
            @test minimum(rho) > 0
            @test maximum(rho) < 4
            p_min = minimum(Trixi.pressure(Trixi.get_node_vars(u, equations, semi.solver, i,
                                                               j, element), equations)
                            for i in 1:3, j in 1:3, element in 1:size(u, 4))
            @test p_min > 0
        end

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end
end

end # module
