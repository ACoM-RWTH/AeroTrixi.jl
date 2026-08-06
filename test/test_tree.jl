module TestExamplesTreeMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "P4estMesh" begin
    @trixi_testset "elixir_eulermst1T_ec.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_eulermst1T_ec.jl"),
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
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
    end
end

end # module