module TestTrixiGridapBridge

using Test
using Trixi
using AeroTrixi
using OrdinaryDiffEqLowStorageRK
# Gridap.jl names are imported explicitly to avoid clashes with Trixi.jl (e.g., `solve`)
import Gridap
using Gridap: ∫, ∇, ⋅, TrialFESpace, AffineFEOperator
using LinearAlgebra: cholesky, Symmetric

const BOX_SIDES = (:x_neg, :x_pos, :y_neg, :y_pos)

# A quadratic function and its gradient; it is in the Q2 space of the five-tree mesh,
# whose elements are straight-sided
quadratic(x) = 1 + 2 * x[1] - 3 * x[2] + 0.5 * x[1]^2 + 1.5 * x[1] * x[2] - x[2]^2
gradient_quadratic(x) = SVector(2 + x[1] + 1.5 * x[2], -3 + 1.5 * x[1] - 2 * x[2])

# Five-tree mesh with all cells in the right half refined once more, which creates hanging
# nodes at the refinement boundary
function five_tree_mesh_with_hanging_nodes()
    mesh = p4est_mesh_five_trees(l = 0.4, polydeg = 2, initial_refinement_level = 1)
    coordinates = Array{Float64}(undef, 2, 2, 2, Trixi.ncells(mesh))
    Trixi.calc_node_coordinates!(coordinates, mesh, [-1.0, 1.0])
    flags = [sum(coordinates[1, :, :, cell]) > 0 ? 1 : 0 for cell in 1:Trixi.ncells(mesh)]
    # Same mechanism as in Trixi.jl's `AMRCallback`
    iter_volume_c = Trixi.cfunction(Trixi.copy_to_quad_iter_volume, Val(2))
    Trixi.iterate_p4est(mesh.p4est, flags; iter_volume_c)
    Trixi.refine!(mesh)
    return mesh
end

@testset "Trixi.jl-Gridap.jl bridge" begin
    @testset "five-tree mesh and Q2 node numbering" begin
        # 20 cells; the Q2 nodes are the vertices of the mesh refined once more: 89 nodes.
        # Without the corner connectivity between the trees, the nodes at the corners of
        # the center square would be duplicated.
        mesh = p4est_mesh_five_trees(l = 0.4, polydeg = 2, initial_refinement_level = 1)
        lnodes = P4estLnodesQ2(mesh)
        @test Trixi.ncells(mesh) == 20
        @test lnodes.n_independent == 89
        @test lnodes.n_hanging == 0
        @test allunique(eachcol(round.(lnodes.coordinates, digits = 12)))
    end

    mesh = five_tree_mesh_with_hanging_nodes()
    lnodes = P4estLnodesQ2(mesh)

    @testset "hanging-node constraints" begin
        @test lnodes.n_hanging > 0
        # The constraints interpolate the coarse face, which is exact for a quadratic
        values = [quadratic(x) for x in eachcol(lnodes.coordinates)]
        constrained = copy(values)
        constrained[(lnodes.n_independent + 1):end] .= NaN
        apply_hanging_constraints!(constrained, lnodes)
        @test constrained ≈ values
    end

    @testset "Q2 Poisson problem with hanging nodes" begin
        # -Δu = 1 for the quadratic u, with Dirichlet values u on the boundary; the Galerkin
        # solution with the hanging-node constraints is exact
        model = gridap_model(lnodes)
        q2 = gridap_q2_space(lnodes, model; dirichlet_tags = BOX_SIDES)
        dΩ = Gridap.Measure(Gridap.Triangulation(model), 4)
        operator = AffineFEOperator((u, v) -> ∫(∇(v) ⋅ ∇(u)) * dΩ, v -> ∫(v * 1.0) * dΩ,
                                    TrialFESpace(q2.space, quadratic), q2.space)
        uh = Gridap.solve(operator)
        node_values = gridap_node_values(uh, q2)
        @test maximum(abs(node_values[k] - quadratic(x))
                      for (k, x) in enumerate(eachcol(lnodes.coordinates))) < 1e-10
    end

    @testset "transfers DG ↔ Q2" begin
        equations = CompressibleEulerPoissonEquations2D(1.4)
        initial_condition = (x, t, equations) -> prim2cons(SVector(quadratic(x), 0.0, 0.0,
                                                                   1.0, 0.0, 0.0, 0.0),
                                                           equations)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                            DGSEM(polydeg = 2);
                                            boundary_conditions = boundary_condition_do_nothing)
        u_ode = Trixi.compute_coefficients(0.0, semi)

        # DG → Q2: the density is continuous, so the averages at shared nodes and the
        # hanging-node constraints reproduce it
        node_values = zeros(1, size(lnodes.coordinates, 2))
        transfer_dg_to_q2!((u, x, equations) -> SVector(u[1]), node_values, u_ode, semi,
                           lnodes)
        @test vec(node_values) ≈ [quadratic(x) for x in eachcol(lnodes.coordinates)]

        # Q2 → DG: the potential φ = density and its field E = -∇φ by injection and with the
        # derivative matrix of Trixi.jl
        transfer_q2_to_dg!(write_electrostatic, u_ode, semi, lnodes, node_values)
        u = Trixi.wrap_array(u_ode, semi)
        x = semi.cache.elements.node_coordinates
        @test u[5, :, :, :] ≈ u[1, :, :, :]
        @test maximum(abs(u[5 + d, i, j, e] + gradient_quadratic(x[:, i, j, e])[d])
                      for d in 1:2, i in 1:3, j in 1:3, e in axes(u, 4)) < 1e-10
    end

    @testset "GridapCallback: order in time with stage updates" begin
        # With the force of the field on the flow, an update of φ and E once per step
        # limits the time integration to first order; the update on each Runge-Kutta stage
        # keeps the fourth order of CarpenterKennedy2N54
        poisson = GridapLinearProblem((φ, v, dΩ) -> ∫(∇(v) ⋅ ∇(φ)) * dΩ,
                                      (v, sources, dΩ) -> ∫(v * sources[1]) * dΩ;
                                      factorize = A -> cholesky(Symmetric(A)))
        initial_condition = function (x, t, equations)
            rho = 1 + 0.5 * exp(-20 * ((x[1] - 0.1)^2 + x[2]^2))
            return prim2cons(SVector(rho, 0.2, -0.1, 1.0, 0.0, 0.0, 0.0), equations)
        end
        function density_at_end(dt, stages)
            equations = CompressibleEulerPoissonEquations2D(1.4; coupling = 5.0)
            mesh = p4est_mesh_five_trees(l = 0.4, trees_per_block = 2, polydeg = 2,
                                         initial_refinement_level = 1)
            semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                                DGSEM(polydeg = 2);
                                                boundary_conditions = boundary_condition_do_nothing,
                                                source_terms = source_terms_electrostatic)
            ode = semidiscretize(semi, (0.0, 0.1))
            callback = GridapCallback(semi, poisson;
                                      source = (u, x, equations) -> SVector(u[1] - 1),
                                      write = write_electrostatic,
                                      dirichlet_tags = BOX_SIDES)
            stage_limiter! = stages ? gridap_stage_update(callback) :
                             OrdinaryDiffEqLowStorageRK.OrdinaryDiffEqCore.trivial_limiter!
            sol = solve(ode,
                        CarpenterKennedy2N54(; stage_limiter!,
                                             williamson_condition = false);
                        dt, adaptive = false, save_everystep = false,
                        callback = CallbackSet(callback))
            return Trixi.wrap_array(sol.u[end], semi)[1, :, :, :]
        end
        reference = density_at_end(0.1 / 160, true)
        for (stages, order) in ((false, 1), (true, 4))
            errors = [maximum(abs, density_at_end(0.1 / n, stages) - reference)
                      for n in (10, 20)]
            @test log2(errors[1] / errors[2])≈order atol=0.2
        end
    end
end

end # module
