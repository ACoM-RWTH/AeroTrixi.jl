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
# 1. Run the circular blast wave on the five-tree mesh with AMR, exactly as in
#    `elixir_euler_blast_wave_five_trees_amr.jl`.
# 2. Take the adapted mesh, create the continuous Q2 node numbering with hanging-node
#    constraints, and the corresponding Gridap.jl model and finite element space.
# 3. Transfer the density deviation ρ - 1 from the DG solution to Gridap.jl and solve the
#    Poisson problem -Δφ = ρ - 1 with φ = 0 on the boundary, using a Cholesky
#    factorization.
# 4. Transfer φ and its gradient back to the DG nodes and plot them.
#
# Besides AeroTrixi.jl and Trixi.jl, this example requires the packages
# OrdinaryDiffEqLowStorageRK.jl, Plots.jl, and Gridap.jl in the active environment.

###############################################################################
# run the blast wave

trixi_include(joinpath(@__DIR__, "elixir_euler_blast_wave_five_trees_amr.jl"))

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

# Density deviation at the DG nodes, interpolated to the Q2 nodes of each element, and
# represented as a discontinuous Q2 function in Gridap.jl
density_deviation = dg_nodal_values((u, equations) -> u[1] - 1, sol.u[end], semi)
source = gridap_dg_function(gridap_dg_q2_space(model),
                            dg_to_q2_elements(density_deviation, solver.basis.nodes))

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
# Gridap → DG: potential and its gradient at the DG nodes

φ_nodes = gridap_node_values(φh, q2)
n_nodes_dg = length(solver.basis.nodes)
φ_dg = zeros(n_nodes_dg, n_nodes_dg, Trixi.ncells(mesh))
q2_to_dg!(φ_dg, φ_nodes, lnodes, solver.basis.nodes)
∇φ_dg = zeros(2, n_nodes_dg, n_nodes_dg, Trixi.ncells(mesh))
q2_gradient_to_dg!(∇φ_dg, φ_nodes, lnodes, semi)

###############################################################################
# plot the density, the potential, and the magnitude of its gradient

# Scalar DG fields on the same mesh are plotted with Trixi.jl's plotting
semi_plot = SemidiscretizationHyperbolic(mesh, LinearScalarAdvectionEquation2D(0.0, 0.0),
                                         (x, t, equations) -> SVector(0.0), solver;
                                         boundary_conditions)
plot_scalar(values) = PlotData2D(vec(values), semi_plot)["scalar"]

pd = PlotData2D(sol)
plot_density = plot(pd["rho"], title = "density at t = $(sol.t[end])")
plot!(plot_density, getmesh(pd), linecolor = :white, linewidth = 0.3)
plot_potential = plot(plot_scalar(φ_dg), title = "potential φ (Q2, Gridap.jl)",
                      color = :balance, clims = (-1, 1) .* maximum(abs, φ_dg))
plot!(plot_potential, getmesh(pd), linecolor = :black, linewidth = 0.2)
gradient_magnitude = sqrt.(∇φ_dg[1, :, :, :] .^ 2 .+ ∇φ_dg[2, :, :, :] .^ 2)
plot_gradient = plot(plot_scalar(gradient_magnitude), title = "|∇φ|")
plot(plot_density, plot_potential, plot_gradient, layout = (1, 3), size = (1600, 450))
savefig(joinpath("out", "blast_five_tree_mesh_gridap_prototype.png"))
