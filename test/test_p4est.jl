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
                                0.05033784140100459,
                                0.05231580880195411,
                                0.05231580880195433,
                                0.291895995407587,
                                0.0002959027211870711,
                                0.0016567690554568433,
                                0.0016567690554568412
                            ],
                            linf=[
                                0.27506990410678345,
                                0.3821085649733141,
                                0.3821085649733142,
                                1.3794900836080721,
                                0.001176085599919377,
                                0.013903874798244082,
                                0.013903874798244129
                            ],
                            tspan=(0.0, 0.05))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end
end

end # module
