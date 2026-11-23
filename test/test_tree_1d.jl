module TestExamplesTreeMesh1D

using Test
using Trixi
using AeroTrixi: flux_ausmplusup

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "TreeMesh1D" begin
    @trixi_testset "elixir_euler_modified_sod_ausmplus.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_modified_sod_ausmplus.jl"),
                            l2=[
                                0.2606060414022102,
                                0.4488446633828044,
                                0.9203592394635516
                            ],
                            linf=[
                                0.5878413702642901,
                                0.8106909192141609,
                                1.7674407034136164
                            ])

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

    @trixi_testset "elixir_navierstokes_convergence_periodic_ausmplus.jl with AUSM+-up" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_navierstokes_convergence_periodic_ausmplus.jl"),
                            surface_flux=flux_ausmplusup,
                            l2=[
                                0.00011262232635716657,
                                4.7089065955241156e-5,
                                0.00031562395112728187
                            ],
                            linf=[
                                0.0005877329087469185,
                                0.00027292465552886114,
                                0.0021671925851407536
                            ])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
        @test_allocations(Trixi.rhs_parabolic!, semi, sol, 1000)
    end
end

end # module
