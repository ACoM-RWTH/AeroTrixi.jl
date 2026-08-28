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

        # Entropy conservation of the semidiscretization: with `flux_oblapenko_etal` used
        # as both the volume and the surface flux, ∑∂S/∂U ⋅ Uₜ vanishes up to
        # round-off. This is a property of the spatial discretization alone, so it
        # holds at any state and does not depend on the time integration scheme.
        let u_ode = sol.u[end]
            du_ode = similar(u_ode)
            Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, sol.t[end])
            dSdt = Trixi.analyze(Trixi.entropy_timederivative,
                                 Trixi.wrap_array(du_ode, semi),
                                 Trixi.wrap_array(u_ode, semi),
                                 sol.t[end], semi)
            @test isapprox(dSdt, 0.0, atol = 1.5e-15)
        end

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end

    @trixi_testset "elixir_eulermst1T_chem.jl" begin
        # `initial_condition` is the analytical solution at time `t`, so these are
        # the errors of the chemistry integration rather than of the flow field
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_eulermst1T_chem.jl"),
                            l2=[
                                3.180301991599695e-15,
                                3.1803019915996924e-15,
                                1.9590192448834736e-14,
                                5.741238820271878e-13,
                                5.693900823875728e-13
                            ],
                            linf=[
                                1.0501327460515194e-14,
                                1.0501327460515076e-14,
                                2.708944180085382e-14,
                                5.748179709996748e-13,
                                5.702660565987117e-13
                            ])

        # The reaction is integrated against its closed-form solution: with constant
        # rate coefficients the species equations decouple from the energy equation,
        # and with `2 n_mol + n_atom` conserved `n_mol` solves a Riccati equation.
        # Roughly 30% of the molecules dissociate over the run.
        let u = Trixi.wrap_array(sol.u[end], semi)
            inv_mass_mol = equations.thermodata.inv_mass[1]
            n_mol = u[4, 1, 1, 1] * inv_mass_mol

            @test isapprox(n_mol, n_mol_exact(sol.t[end]), rtol = 1e-10)
            @test isapprox(n_mol, 0.35, rtol = 1e-2)

            # the state stays uniform: the DG operator contributes nothing here, so
            # every node must follow the same reaction ODE
            n_mol_nodes = @view(u[4, :, :, :]) .* inv_mass_mol
            @test maximum(abs.(n_mol_nodes .- n_mol)) < 1e-13
        end

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end
end

end # module
