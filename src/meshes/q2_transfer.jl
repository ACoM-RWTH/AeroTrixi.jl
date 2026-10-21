# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing-EC-Trixi for further details.
@muladd begin
#! format: noindent

# Transfer of nodal data between Trixi.jl's DG elements with `polydeg = 2` and the
# continuous Q2 nodes of a `P4estLnodesQ2`. Both use the same elements, and the DG nodes
# (-1, 0, 1) of each element coincide with its Q2 nodes, so that the transfers are simple
# node-to-node operations:
# - DG → Q2: each Q2 node gets the average of the DG values of all elements sharing it;
#   the hanging nodes then get the values of the coarse faces they lie on,
# - Q2 → DG: each DG node gets the value of its Q2 node (injection).

function check_q2_dg_nodes(dg_values, lnodes)
    @assert size(dg_values, ndims(dg_values))==size(lnodes.element_nodes, 2) "number of elements does not match"
    @assert size(dg_values, ndims(dg_values) - 1)==
            size(dg_values, ndims(dg_values) - 2)==3 "DG data must have polydeg = 2"
    return nothing
end

function check_q2_semi(semi)
    nodes = mesh_equations_solver_cache(semi)[3].basis.nodes
    @assert length(nodes) == 3&&nodes ≈ [-1, 0, 1] "the transfer requires polydeg = 2"
    return nothing
end

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
    dg_to_q2!(node_values, dg_values, lnodes::P4estLnodesQ2)

Transfer DG nodal data of size `(3, 3, n_elements)` (DG with `polydeg = 2`) to a continuous
Q2 field at the nodes of `lnodes`. Each independent node gets the average of the DG values
at this node of all elements sharing it. The hanging nodes then get the values of the
quadratic on the coarse face they lie on, interpolated from the averaged values at the
nodes of this face (see [`apply_hanging_constraints!`](@ref)), so that the field is
continuous across hanging faces.
"""
function dg_to_q2!(node_values, dg_values::AbstractArray{<:Any, 3},
                   lnodes::P4estLnodesQ2)
    check_q2_dg_nodes(dg_values, lnodes)
    @assert length(node_values)==n_nodes(lnodes) "node_values must contain all nodes"
    element_nodes = lnodes.element_nodes
    counts = zeros(Int, length(node_values))
    fill!(node_values, zero(eltype(node_values)))
    for element in axes(dg_values, 3), j in 1:3, i in 1:3
        node = element_nodes[i + 3 * (j - 1), element]
        node_values[node] += dg_values[i, j, element]
        counts[node] += 1
    end
    node_values ./= counts
    # Values at the hanging nodes from the coarse faces
    apply_hanging_constraints!(node_values, lnodes)
    return node_values
end

"""
    q2_to_dg!(dg_values, node_values, lnodes::P4estLnodesQ2)

Inject the values at the nodes of `lnodes` into DG nodal data of size `(3, 3, n_elements)`
(DG with `polydeg = 2`): each DG node gets the value of its Q2 node.
"""
function q2_to_dg!(dg_values::AbstractArray{<:Any, 3}, node_values,
                   lnodes::P4estLnodesQ2)
    check_q2_dg_nodes(dg_values, lnodes)
    @assert length(node_values)==n_nodes(lnodes) "node_values must contain all nodes"
    element_nodes = lnodes.element_nodes
    for element in axes(dg_values, 3), j in 1:3, i in 1:3
        dg_values[i, j, element] = node_values[element_nodes[i + 3 * (j - 1), element]]
    end
    return dg_values
end

"""
    q2_gradient_to_dg!(dg_gradients, node_values, lnodes::P4estLnodesQ2, semi)

Physical gradient of the Q2 function with values `node_values` at the DG nodes of `semi`
(DG with `polydeg = 2`), of size `(2, 3, 3, n_elements)`. It is the local derivative of
the injected values within each element, computed with the derivative matrix and metric
terms of Trixi.jl, and discontinuous between elements.
"""
function q2_gradient_to_dg!(dg_gradients::AbstractArray{<:Any, 4}, node_values,
                            lnodes::P4estLnodesQ2, semi)
    check_q2_dg_nodes(dg_gradients, lnodes)
    check_q2_semi(semi)
    @assert length(node_values)==n_nodes(lnodes) "node_values must contain all nodes"
    _, _, solver, cache = mesh_equations_solver_cache(semi)
    for element in axes(dg_gradients, 4), j in 1:3, i in 1:3
        gradient = q2_gradient_at_node(node_values, lnodes, solver, cache, i, j,
                                       element)
        dg_gradients[1, i, j, element] = gradient[1]
        dg_gradients[2, i, j, element] = gradient[2]
    end
    return dg_gradients
end

"""
    q2_to_dg!(func, u_ode, semi, node_values, lnodes::P4estLnodesQ2)

Write a Q2 function with values `node_values` at the nodes of `lnodes` into the solution
vector `u_ode` of `semi` (DG with `polydeg = 2`). At each DG node, the new state is
`func(u_node, φ, ∇φ, equations)`, where `u_node` is the current state, `φ` the injected
value, and `∇φ` the physical gradient as in [`q2_gradient_to_dg!`](@ref). For example,
```julia
q2_to_dg!(u_ode, semi, φ_nodes, lnodes) do u, φ, ∇φ, equations
    return prim2cons(SVector(u[1], ∇φ[1], ∇φ[2], φ), equations)
end
```
keeps the density and sets the velocity to `∇φ` and the pressure to `φ`.
"""
function q2_to_dg!(func, u_ode, semi, node_values, lnodes::P4estLnodesQ2)
    check_q2_semi(semi)
    @assert length(node_values)==n_nodes(lnodes) "node_values must contain all nodes"
    _, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, semi)
    @assert size(u, 4)==size(lnodes.element_nodes, 2) "number of elements does not match"
    element_nodes = lnodes.element_nodes
    for element in axes(u, 4), j in 1:3, i in 1:3
        φ = node_values[element_nodes[i + 3 * (j - 1), element]]
        ∇φ = q2_gradient_at_node(node_values, lnodes, solver, cache, i, j, element)
        u_node = get_node_vars(u, equations, solver, i, j, element)
        set_node_vars!(u, func(u_node, φ, ∇φ, equations), equations, solver, i, j,
                       element)
    end
    return u_ode
end

# Physical gradient of the Q2 function at DG node `(i, j)` of `element`, using the
# derivative matrix and metric terms of Trixi.jl
@inline function q2_gradient_at_node(node_values, lnodes, solver, cache, i, j, element)
    @unpack contravariant_vectors, inverse_jacobian = cache.elements
    element_nodes = lnodes.element_nodes
    D = solver.basis.derivative_matrix
    # Derivatives with respect to the reference coordinates
    dξ = zero(eltype(node_values))
    dη = zero(eltype(node_values))
    for k in 1:3
        dξ += D[i, k] * node_values[element_nodes[k + 3 * (j - 1), element]]
        dη += D[j, k] * node_values[element_nodes[i + 3 * (k - 1), element]]
    end
    # ∂/∂x_d = (Ja^1_d ∂/∂ξ + Ja^2_d ∂/∂η) / J
    return inverse_jacobian[i, j, element] *
           SVector(contravariant_vectors[1, 1, i, j, element] * dξ +
                   contravariant_vectors[1, 2, i, j, element] * dη,
                   contravariant_vectors[2, 1, i, j, element] * dξ +
                   contravariant_vectors[2, 2, i, j, element] * dη)
end
end # @muladd
