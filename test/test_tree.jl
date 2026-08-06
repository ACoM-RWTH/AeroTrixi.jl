module TestExamplesTreeMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "TreeMesh2D" begin
    @trixi_testset "elixir_eulermst1T_ec.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_eulermst1T_ec.jl"),
                            l2=[
                                0.04489467251664586,
                                0.04489632946302154,
                                0.3210538730770086,
                                0.03296430184879192,
                                0.01648215092439596
                            ],
                            linf=[
                                0.3006632724270793,
                                0.30066009586298514,
                                1.471289021801013,
                                0.14959425977427443,
                                0.07479712988713721
                            ],
                            tspan=(0.0, 0.1))

        # Entropy conservation of the semidiscretization: with `flux_oblapenko` used
        # as both the volume and the surface flux, ∑∂S/∂U ⋅ Uₜ vanishes up to
        # round-off. This is a property of the spatial discretization alone, so it
        # holds at any state and does not depend on the time integration scheme.
        let u_ode = sol.u[end]
            du_ode = similar(u_ode)
            Trixi.rhs!(du_ode, u_ode, semi, sol.t[end])
            dSdt = Trixi.analyze(Trixi.entropy_timederivative,
                                 Trixi.wrap_array(du_ode, semi),
                                 Trixi.wrap_array(u_ode, semi),
                                 sol.t[end], semi)
            @test isapprox(dSdt, 0.0, atol = 1e-15)
        end

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs!, semi, sol, 1000)
    end
end

end # module
