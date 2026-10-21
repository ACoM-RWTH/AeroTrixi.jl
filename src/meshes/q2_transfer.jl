# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing-EC-Trixi for further details.
@muladd begin
#! format: noindent

# Transfer of nodal data between Trixi.jl's DG elements and the continuous Q2 nodes of a
# `P4estLnodesQ2`. Both use the same elements, and the Q2 nodes of an element are the
# tensor-product nodes (-1, 0, 1) in the reference coordinates of the DG element. With
# `polydeg = 2`, they coincide with the DG nodes and all transfers are exact.

# Lagrange polynomials of the nodes (-1, 0, 1) and their derivatives
@inline q2_basis(ξ) = SVector(ξ * (ξ - 1) / 2, 1 - ξ^2, ξ * (ξ + 1) / 2)
@inline q2_basis_derivative(ξ) = SVector(ξ - 0.5, -2 * ξ, ξ + 0.5)

"""
    dg_nodal_values(func, u_ode, semi)

Evaluate `func(u_node, equations)` at all DG nodes, e.g., `func = density`. Return an
array of size `(n_nodes, n_nodes, n_elements)`, where `n_nodes = polydeg + 1`.
"""
function dg_nodal_values(func, u_ode, semi)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, semi)
    n_nodes = nnodes(solver)
    values = Array{real(solver)}(undef, n_nodes, n_nodes, size(u, 4))
    for element in axes(u, 4), j in 1:n_nodes, i in 1:n_nodes
        u_node = get_node_vars(u, equations, solver, i, j, element)
        values[i, j, element] = func(u_node, equations)
    end
    return values
end

"""
    dg_to_q2_elements(dg_values, dg_nodes)

Interpolate DG nodal data of size `(n_nodes, n_nodes, n_elements)` at the tensor-product
nodes `dg_nodes` of each element to the nine Q2 nodes of each element. Return a
`9 × n_elements` matrix in the lexicographic node order of `P4estLnodesQ2.element_nodes`.

The result is discontinuous between elements, like the DG data. It is exact for DG data of
`polydeg = 2` and for polynomials of degree up to 2 in each reference coordinate.
"""
function dg_to_q2_elements(dg_values::AbstractArray{<:Any, 3}, dg_nodes)
    n_nodes = length(dg_nodes)
    @assert size(dg_values, 1)==size(dg_values, 2)==n_nodes "dg_values do not match dg_nodes"
    # Interpolation from the DG nodes to the Q2 nodes (-1, 0, 1)
    M = polynomial_interpolation_matrix(collect(dg_nodes), [-1.0, 0.0, 1.0])
    n_elements = size(dg_values, 3)
    element_values = zeros(eltype(dg_values), 9, n_elements)
    for element in 1:n_elements, j in 1:3, i in 1:3
        value = zero(eltype(dg_values))
        for jj in 1:n_nodes, ii in 1:n_nodes
            value += M[i, ii] * M[j, jj] * dg_values[ii, jj, element]
        end
        element_values[i + 3 * (j - 1), element] = value
    end
    return element_values
end

"""
    q2_to_dg!(dg_values, node_values, lnodes::P4estLnodesQ2, dg_nodes)

Evaluate the continuous Q2 function with values `node_values` at the nodes of `lnodes`
(including hanging nodes) at the tensor-product nodes `dg_nodes` of each element.
`dg_values` has size `(n_nodes, n_nodes, n_elements)`.
"""
function q2_to_dg!(dg_values::AbstractArray{<:Any, 3}, node_values,
                   lnodes::P4estLnodesQ2, dg_nodes)
    element_nodes = lnodes.element_nodes
    @assert length(node_values)==n_nodes(lnodes) "node_values must contain all nodes"
    @assert size(dg_values, 3)==size(element_nodes, 2) "number of elements does not match"
    basis = map(q2_basis, dg_nodes)
    for element in axes(dg_values, 3), j in eachindex(dg_nodes),
        i in eachindex(dg_nodes)

        value = zero(eltype(dg_values))
        for jj in 1:3, ii in 1:3
            node = element_nodes[ii + 3 * (jj - 1), element]
            value += basis[i][ii] * basis[j][jj] * node_values[node]
        end
        dg_values[i, j, element] = value
    end
    return dg_values
end

"""
    q2_gradient_to_dg!(dg_gradients, node_values, lnodes::P4estLnodesQ2, semi)

Evaluate the physical gradient of the continuous Q2 function with values `node_values` at
the DG nodes of `semi`, using the metric terms of Trixi.jl. `dg_gradients` has size
`(2, n_nodes, n_nodes, n_elements)`. Within each element, the gradient is the exact
derivative of the Q2 polynomial; it is discontinuous between elements.
"""
function q2_gradient_to_dg!(dg_gradients::AbstractArray{<:Any, 4}, node_values,
                            lnodes::P4estLnodesQ2, semi)
    _, _, solver, cache = mesh_equations_solver_cache(semi)
    @unpack contravariant_vectors, inverse_jacobian = cache.elements
    element_nodes = lnodes.element_nodes
    @assert length(node_values)==n_nodes(lnodes) "node_values must contain all nodes"
    @assert size(dg_gradients, 4)==size(element_nodes, 2) "number of elements does not match"
    dg_nodes = solver.basis.nodes
    basis = map(q2_basis, dg_nodes)
    derivative = map(q2_basis_derivative, dg_nodes)
    for element in axes(dg_gradients, 4), j in eachindex(dg_nodes),
        i in eachindex(dg_nodes)

        # Derivatives with respect to the reference coordinates
        dξ = zero(eltype(dg_gradients))
        dη = zero(eltype(dg_gradients))
        for jj in 1:3, ii in 1:3
            value = node_values[element_nodes[ii + 3 * (jj - 1), element]]
            dξ += derivative[i][ii] * basis[j][jj] * value
            dη += basis[i][ii] * derivative[j][jj] * value
        end
        # ∂/∂x_d = (Ja^1_d ∂/∂ξ + Ja^2_d ∂/∂η) / J
        for d in 1:2
            dg_gradients[d, i, j, element] = inverse_jacobian[i, j, element] *
                                             (contravariant_vectors[d, 1, i, j,
                                                                    element] *
                                              dξ +
                                              contravariant_vectors[d, 2, i, j,
                                                                    element] *
                                              dη)
        end
    end
    return dg_gradients
end
end # @muladd
