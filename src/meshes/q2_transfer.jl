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
# - Q2 → DG: each DG node gets the value of its Q2 node (injection); local derivatives are
#   computed within each element.
# Data of several fields at the Q2 nodes are stored as matrices `n_fields × n_nodes`.

function check_q2_semi(semi, lnodes)
    _, _, solver, cache = mesh_equations_solver_cache(semi)
    nodes = solver.basis.nodes
    @assert length(nodes) == 3&&nodes ≈ [-1, 0, 1] "the transfer requires polydeg = 2"
    @assert nelements(solver, cache)==size(lnodes.element_nodes, 2) "number of elements does not match"
    return nothing
end

"""
    transfer_dg_to_q2!(func, node_values, u_ode, semi, lnodes::P4estLnodesQ2)

Transfer quantities computed from the DG solution `u_ode` of `semi` (DG with
`polydeg = 2`) to a continuous Q2 field at the nodes of `lnodes`.

At each DG node, `func(u_node, x, equations)` returns an `SVector` with one entry per
field, e.g., `SVector(u_node[1] - 1)` for a single field. `node_values` is a matrix of size
`n_fields × n_nodes`. Each independent node gets the average of the values of all
elements sharing it. The hanging nodes then get the values of the quadratic on the coarse
face they lie on (see [`apply_hanging_constraints!`](@ref)), so that the fields are
continuous.
"""
function transfer_dg_to_q2!(func, node_values::AbstractMatrix, u_ode, semi,
                            lnodes::P4estLnodesQ2)
    check_q2_semi(semi, lnodes)
    @assert size(node_values, 2)==n_nodes(lnodes) "node_values must have a column per node"
    _, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, semi)
    node_coordinates = cache.elements.node_coordinates
    element_nodes = lnodes.element_nodes
    n_fields = size(node_values, 1)

    counts = zeros(Int, size(node_values, 2))
    fill!(node_values, zero(eltype(node_values)))
    for element in axes(u, 4), j in 1:3, i in 1:3
        node = element_nodes[i + 3 * (j - 1), element]
        u_node = get_node_vars(u, equations, solver, i, j, element)
        x = get_node_coords(node_coordinates, equations, solver, i, j, element)
        values = func(u_node, x, equations)
        @assert length(values)==n_fields "func must return one value per field"
        for field in 1:n_fields
            node_values[field, node] += values[field]
        end
        counts[node] += 1
    end
    for node in axes(node_values, 2), field in 1:n_fields
        node_values[field, node] /= counts[node]
    end

    # Values at the hanging nodes from the coarse faces
    apply_hanging_constraints!(node_values, lnodes)
    return node_values
end

"""
    Q2NodeData

Q2 data at a DG node, passed to the function of [`transfer_q2_to_dg!`](@ref). Use
[`q2_value`](@ref) and [`q2_gradient`](@ref) to access the value and the gradient of a
field at the node.
"""
struct Q2NodeData{NodeValues, DerivativeMatrix, ContravariantVectors, InverseJacobian}
    node_values           :: NodeValues
    element_nodes         :: SVector{9, Int}
    i                     :: Int
    j                     :: Int
    element               :: Int
    derivative_matrix     :: DerivativeMatrix
    contravariant_vectors :: ContravariantVectors
    inverse_jacobian      :: InverseJacobian
end

"""
    q2_value(q::Q2NodeData, field = 1)

Value of the Q2 field `field` at the DG node (injection).
"""
@inline function q2_value(q::Q2NodeData, field = 1)
    return q.node_values[field, q.element_nodes[q.i + 3 * (q.j - 1)]]
end

"""
    q2_gradient(q::Q2NodeData, field = 1)

Physical gradient of the Q2 field `field` at the DG node, i.e., the local derivative within
the element, computed with the derivative matrix and metric terms of Trixi.jl. It is
discontinuous between elements and only computed when called.
"""
@inline function q2_gradient(q::Q2NodeData, field = 1)
    @unpack node_values, element_nodes, i, j, element = q
    D = q.derivative_matrix
    # Derivatives with respect to the reference coordinates
    dξ = zero(eltype(node_values))
    dη = zero(eltype(node_values))
    @inbounds for k in 1:3
        dξ += D[i, k] * node_values[field, element_nodes[k + 3 * (j - 1)]]
        dη += D[j, k] * node_values[field, element_nodes[i + 3 * (k - 1)]]
    end
    # ∂/∂x_d = (Ja^1_d ∂/∂ξ + Ja^2_d ∂/∂η) / J
    Ja = q.contravariant_vectors
    return @inbounds q.inverse_jacobian[i, j, element] *
                     SVector(Ja[1, 1, i, j, element] * dξ +
                             Ja[1, 2, i, j, element] * dη,
                             Ja[2, 1, i, j, element] * dξ +
                             Ja[2, 2, i, j, element] * dη)
end

"""
    transfer_q2_to_dg!(func, u_ode, semi, lnodes::P4estLnodesQ2, node_values)

Write data computed from Q2 fields at the nodes of `lnodes` into the DG solution `u_ode` of
`semi` (DG with `polydeg = 2`). `node_values` is a vector (a single field) or a matrix
`n_fields × n_nodes`.

At each DG node, the new state is `func(u_node, q, x, equations)`, where `u_node` is the
current state and `q` gives access to the Q2 fields at this node with [`q2_value`](@ref)
and [`q2_gradient`](@ref). For example,
```julia
transfer_q2_to_dg!(u_ode, semi, lnodes, φ_nodes) do u, q, x, equations
    φ = q2_value(q)
    ∇φ = q2_gradient(q)
    return prim2cons(SVector(u[1], ∇φ[1], ∇φ[2], φ), equations)
end
```
keeps the density and sets the velocity to `∇φ` and the pressure to `φ`.
"""
function transfer_q2_to_dg!(func, u_ode, semi, lnodes::P4estLnodesQ2,
                            node_values::AbstractVector)
    return transfer_q2_to_dg!(func, u_ode, semi, lnodes, reshape(node_values, 1, :))
end

function transfer_q2_to_dg!(func, u_ode, semi, lnodes::P4estLnodesQ2,
                            node_values::AbstractMatrix)
    check_q2_semi(semi, lnodes)
    @assert size(node_values, 2)==n_nodes(lnodes) "node_values must have a column per node"
    _, equations, solver, cache = mesh_equations_solver_cache(semi)
    @unpack node_coordinates, contravariant_vectors, inverse_jacobian = cache.elements
    derivative_matrix = solver.basis.derivative_matrix
    u = wrap_array(u_ode, semi)
    element_nodes = lnodes.element_nodes
    for element in axes(u, 4)
        local_nodes = SVector{9, Int}(ntuple(k -> element_nodes[k, element], Val(9)))
        for j in 1:3, i in 1:3
            q = Q2NodeData(node_values, local_nodes, i, j, element, derivative_matrix,
                           contravariant_vectors, inverse_jacobian)
            u_node = get_node_vars(u, equations, solver, i, j, element)
            x = get_node_coords(node_coordinates, equations, solver, i, j, element)
            set_node_vars!(u, func(u_node, q, x, equations), equations, solver, i, j,
                           element)
        end
    end
    return u_ode
end
end # @muladd
