using OrdinaryDiffEqLowStorageRK
using Trixi
using AeroTrixi
using Plots
using LinearAlgebra: cholesky, Symmetric

###############################################################################
# Prototype of the coupling between Trixi.jl and Gridap.jl.
#
# 1. Run the circular blast wave on the five-tree mesh with AMR, exactly as in
#    `elixir_euler_blast_wave_five_trees_amr.jl`.
# 2. Take the adapted mesh, create the continuous Q2 node numbering with hanging-node
#    constraints, and the corresponding Gridap.jl model and finite element space.
# 3. Solve the Poisson problem -Δφ = f with φ = 0 on the boundary, using a Cholesky
#    factorization, and plot φ on the adapted mesh.
#
# Besides AeroTrixi.jl and Trixi.jl, this example requires the packages
# OrdinaryDiffEqLowStorageRK.jl, Plots.jl, and Gridap.jl in the active environment.

###############################################################################
# run the blast wave

trixi_include(joinpath(@__DIR__, "elixir_euler_blast_wave_five_trees_amr.jl"))

###############################################################################
# continuous Q2 discretization on the adapted mesh

# Gridap.jl is loaded only now and with explicit imports, since it exports names such
# as `solve` that clash with the packages used for the time integration above
using Gridap: Gridap, Triangulation, Measure, TrialFESpace, AffineFEOperator, FEFunction,
              get_matrix, get_vector, num_free_dofs, interpolate, get_free_dof_values,
              get_dirichlet_dof_values, ∫, ∇, ⋅

# Node numbering with hanging nodes and their constraints, derived from `p4est_lnodes`
lnodes = P4estLnodesQ2(mesh)

# Gridap model with the boundary names of the mesh as tags
model = gridap_model(lnodes)

# Q2 space with homogeneous Dirichlet conditions on all sides of the box
q2 = gridap_q2_space(lnodes, model; dirichlet_tags = (:x_neg, :x_pos, :y_neg, :y_pos))
V = q2.space
U = TrialFESpace(V, 0.0)

###############################################################################
# Poisson problem -Δφ = f, φ = 0 on the boundary

# A positive and a negative Gaussian source
function source(x)
    return 50 * exp(-40 * ((x[1] - 0.3)^2 + (x[2] - 0.2)^2)) -
           30 * exp(-60 * ((x[1] + 0.4)^2 + (x[2] + 0.3)^2))
end

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
# plot the density of the blast wave and the potential on the adapted mesh

# Values of φ at all Q2 nodes, including the hanging nodes
φ_nodes = interpolate(φh, q2.unconstrained)
free_values = get_free_dof_values(φ_nodes)
dirichlet_values = get_dirichlet_dof_values(φ_nodes)
node_values = [dof > 0 ? free_values[dof] : dirichlet_values[-dof]
               for dof in q2.node_to_dof]

# For plotting, interpolate the Q2 polynomial of each element to the nodes of a Trixi.jl
# scalar field on the same mesh; the elements of Trixi.jl and Gridap.jl coincide
equations_plot = LinearScalarAdvectionEquation2D(0.0, 0.0)
semi_plot = SemidiscretizationHyperbolic(mesh, equations_plot,
                                         (x, t, equations) -> SVector(0.0),
                                         DGSEM(polydeg = Trixi.polydeg(solver));
                                         boundary_conditions)
lagrange_q2(ξ) = SVector(ξ * (ξ - 1) / 2, 1 - ξ^2, ξ * (ξ + 1) / 2) # nodes -1, 0, 1
nodes_plot = semi_plot.solver.basis.nodes
L = reduce(vcat, (lagrange_q2(ξ)' for ξ in nodes_plot))
φ_plot = zeros(1, length(nodes_plot), length(nodes_plot), Trixi.ncells(mesh))
for element in axes(φ_plot, 4)
    element_values = reshape(node_values[lnodes.element_nodes[:, element]], 3, 3)
    φ_plot[1, :, :, element] .= L * element_values * L'
end

pd = PlotData2D(sol)
plot_density = plot(pd["rho"], title = "density at t = $(sol.t[end])")
plot!(plot_density, getmesh(pd), linecolor = :white, linewidth = 0.3)
pd_φ = PlotData2D(vec(φ_plot), semi_plot)
plot_potential = plot(pd_φ["scalar"], title = "Poisson solution φ (Q2, Gridap.jl)",
                      color = :balance, clims = (-1, 1) .* maximum(abs, φ_plot))
plot!(plot_potential, getmesh(pd), linecolor = :black, linewidth = 0.2)
plot(plot_density, plot_potential, layout = (1, 2), size = (1100, 450))
savefig(joinpath("out", "blast_five_tree_mesh_gridap_prototype.png"))
