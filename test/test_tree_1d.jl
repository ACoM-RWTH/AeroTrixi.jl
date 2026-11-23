module TestExamplesTreeMesh1D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "TreeMesh1D" begin
    @trixi_testset "elixir_euler_sedov_blast_wave_ausmplus.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_sedov_blast_wave_ausmplus.jl"),
                            l2=[
                                0.8628124802061918,
                                0.557396918783315,
                                0.9099876895134151
                            ],
                            linf=[
                                3.278024820231943,
                                2.2888170680360536,
                                2.4304337465927164
                            ],
                            tspan=(0.0, 1.0))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end

    @trixi_testset "elixir_navierstokes_convergence_periodic_ausmplus.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_navierstokes_convergence_periodic_ausmplus.jl"),
                            l2=[
                                0.0001188894409916736,
                                5.493515697375269e-5,
                                0.0005458134837303386
                            ],
                            linf=[
                                0.0004961642897844953,
                                0.0002628104038207013,
                                0.0038776069262151935
                            ])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end
end

end # module
