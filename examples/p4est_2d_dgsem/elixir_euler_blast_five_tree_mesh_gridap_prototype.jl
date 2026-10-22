using OrdinaryDiffEqLowStorageRK
using Trixi
using AeroTrixi
using Plots
using LinearAlgebra: cholesky, Symmetric

# Gridap.jl exports names such as `solve` that clash with the packages used for the time
# integration. `import Gridap` loads the package (and the Gridap.jl extension of
# AeroTrixi.jl) without bringing its names into scope; only the names needed below are
# imported explicitly.
import Gridap
using Gridap: Triangulation, Measure, TrialFESpace, AffineFEOperator, FEFunction,
              get_matrix, get_vector, num_free_dofs, ∫, ∇, ⋅

###############################################################################
# Prototype of the coupling between Trixi.jl and Gridap.jl.
#
# 1. Run the circular blast wave on the five-tree mesh with AMR as in
#    `elixir_euler_blast_wave_five_trees_amr.jl`, but with polynomial degree 2, so that
#    the DG nodes of each element coincide with its Q2 nodes.
# 2. Take the adapted mesh, create the continuous Q2 node numbering with hanging-node
#    constraints, and the corresponding Gridap.jl model and finite element space.
# 3. Transfer the density deviation ρ - 1 from the DG nodes to the Q2 nodes (average of
#    the elements sharing a node) and solve the Poisson problem -Δφ = ρ - 1 with φ = 0 on
#    the boundary, using a Cholesky factorization.
# 4. Write φ and its gradient back into the Euler solution vector, as pressure and
#    velocity, keeping the density, and plot the solution.
#
# Besides AeroTrixi.jl and Trixi.jl, this example requires the packages
# OrdinaryDiffEqLowStorageRK.jl, Plots.jl, and Gridap.jl in the active environment.

###############################################################################
# run the blast wave

trixi_include(joinpath(@__DIR__, "elixir_euler_blast_wave_five_trees_amr.jl"),
              tspan = (0.0, 0.3),
              basis = LobattoLegendreBasis(2),
              mesh = p4est_mesh_five_trees(l = 0.4, trees_per_block = 2, polydeg = 2,
                                           initial_refinement_level = 1))

###############################################################################
# continuous Q2 discretization on the adapted mesh

# Node numbering with hanging nodes and their constraints, derived from `p4est_lnodes`
lnodes = P4estLnodesQ2(mesh)

# Gridap model with the boundary names of the mesh as tags
model = gridap_model(lnodes)

# Q2 space with homogeneous Dirichlet conditions on all sides of the box
q2 = gridap_q2_space(lnodes, model; dirichlet_tags = (:x_neg, :x_pos, :y_neg, :y_pos))
V = q2.space
U = TrialFESpace(V, 0.0)

###############################################################################
# DG → Gridap: density deviation as source term

# Density deviation at the DG nodes, averaged at the Q2 nodes shared by several elements
density_deviation = dg_nodal_values((u, equations) -> u[1] - 1, sol.u[end], semi)
source_nodes = dg_to_q2!(zeros(length(q2.node_to_dof)), density_deviation, lnodes)
source = gridap_q2_function(source_nodes, q2)

###############################################################################
# Poisson problem -Δφ = ρ - 1, φ = 0 on the boundary

Ω = Triangulation(model)
dΩ = Measure(Ω, 4)
a(φ, v) = ∫(∇(v) ⋅ ∇(φ)) * dΩ
l(v) = ∫(v * source) * dΩ

operator = AffineFEOperator(a, l, U, V)
A = get_matrix(operator)
b = get_vector(operator)

# The matrix is symmetric positive definite, also with the hanging-node constraints
factorization = cholesky(Symmetric(A))
φh = FEFunction(U, factorization \ b)

println("Poisson problem: ", lnodes, ", ", num_free_dofs(V), " free degrees of freedom")

###############################################################################
# Gridap → DG: write φ and ∇φ into the Euler solution

# Values of φ at all Q2 nodes, including the hanging nodes
φ_nodes = gridap_node_values(φh, q2)

# At each DG node, φ is injected and ∇φ is the local derivative. For demonstration, the
# final blast wave solution is overwritten: the density is kept, the velocity is set to
# ∇φ, and the pressure to φ.
q2_to_dg!(sol.u[end], semi, φ_nodes, lnodes) do u, φ, ∇φ, equations
    return prim2cons(SVector(u[1], ∇φ[1], ∇φ[2], φ), equations)
end

###############################################################################
# plot the density, the potential (pressure), and its gradient (velocity)

pd = PlotData2D(sol)
plot_density = plot(pd["rho"], title = "density at t = $(sol.t[end])")
plot!(plot_density, getmesh(pd), linecolor = :white, linewidth = 0.3)
plot_potential = plot(pd["p"], title = "pressure ← φ (Q2, Gridap.jl)", color = :balance,
                      clims = (-1, 1) .* maximum(abs, φ_nodes))
plot!(plot_potential, getmesh(pd), linecolor = :black, linewidth = 0.2)
plot_v1 = plot(pd["v1"], title = "v1 ← ∂φ/∂x", color = :balance)
plot_v2 = plot(pd["v2"], title = "v2 ← ∂φ/∂y", color = :balance)
plot(plot_density, plot_potential, plot_v1, plot_v2, layout = (2, 2), size = (1100, 900))
savefig(joinpath("out", "blast_five_tree_mesh_gridap_prototype.png"))
