# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing-EC-Trixi for further details.
@muladd begin
#! format: noindent

"""
    P4estLnodesQ2(mesh::P4estMesh{2})

Continuous Q2 node numbering of the current forest of a 2D `P4estMesh`, including hanging
nodes, their linear constraints, and boundary labels. It is derived from
`p4est_lnodes_new(p4est, ghost, 2)` and must be rebuilt whenever the mesh changes.

The nodes of each element are numbered like the nodes of a Trixi.jl element with
`polydeg = 2`, i.e., in lexicographic order of the reference coordinates `(-1, 0, 1)` with
the first coordinate running fastest. All data are stored on the Julia side; the `p4est`
structures are freed after construction. Only serial runs are supported.

# Fields
- `element_nodes::Matrix{Int}`: `9 × n_elements` node ids of each element, where
  `1:n_independent` are independent nodes and `n_independent .+ (1:n_hanging)` are
  hanging nodes.
- `coordinates::Matrix{Float64}`: `2 × n_nodes` node coordinates.
- `n_independent::Int`, `n_hanging::Int`: number of independent and hanging nodes.
- `hanging_masters::Matrix{Int}`: `3 × n_hanging` independent master nodes `(a, m, b)` of
  each hanging node, where `a` and `b` are the end points of the coarse face and `m` is
  its midpoint. Each hanging node lies between `a` and `m`, at a quarter of the face.
- `hanging_coefficients::Matrix{Float64}`: `3 × n_hanging` constraint coefficients; the
  value of hanging node `n_independent + k` is `sum(hanging_coefficients[:, k] .*
  values[hanging_masters[:, k]])`.
- `boundary_faces::Vector{Tuple{Int, Int, Symbol}}`: `(element, face, name)` of each
  element face on the boundary, with faces numbered as in `p4est` but one-based.
- `boundary_entities::Vector{Vector{Symbol}}`: distinct sorted sets of boundary names
  carried by nodes. A node at the junction of two boundaries, e.g., a corner of the
  domain, belongs to an entity with both names.
- `node_entity::Vector{Int}`: index into `boundary_entities` for each node, `0` for
  interior nodes.

See also [`apply_hanging_constraints!`](@ref).
"""
struct P4estLnodesQ2
    element_nodes        :: Matrix{Int}
    coordinates          :: Matrix{Float64}
    n_independent        :: Int
    n_hanging            :: Int
    hanging_masters      :: Matrix{Int}
    hanging_coefficients :: Matrix{Float64}
    boundary_faces       :: Vector{Tuple{Int, Int, Symbol}}
    boundary_entities    :: Vector{Vector{Symbol}}
    node_entity          :: Vector{Int}
end

n_nodes(lnodes::P4estLnodesQ2) = lnodes.n_independent + lnodes.n_hanging

function Base.show(io::IO, lnodes::P4estLnodesQ2)
    print(io, "P4estLnodesQ2(", size(lnodes.element_nodes, 2), " elements, ",
          lnodes.n_independent, " independent nodes, ", lnodes.n_hanging,
          " hanging nodes)")
    return nothing
end

# Node slots (zero-based, lexicographic) of the corners and face midpoints of a Q2 element.
# Corners and faces are numbered as in `p4est`.
const Q2_CORNER_SLOTS = (0, 2, 6, 8)
const Q2_FACE_MIDPOINT_SLOTS = (3, 5, 1, 7)
const Q2_FACE_SLOTS = ((0, 3, 6), (2, 5, 8), (0, 1, 2), (6, 7, 8))

# Values of the quadratic Lagrange polynomials of the coarse face (nodes a, m, b at
# s = 0, 1/2, 1) at s = 1/4
const Q2_HANGING_COEFFICIENTS = (3 / 8, 3 / 4, -1 / 8)

function P4estLnodesQ2(mesh::P4estMesh{2})
    if mpi_isparallel()
        error("P4estLnodesQ2 is only implemented for serial runs")
    end

    element_nodes_p4est, face_codes = lnodes_q2_from_p4est(mesh)
    n_elements = size(element_nodes_p4est, 2)
    @assert n_elements==ncells(mesh) "number of lnodes elements does not match the mesh"
    n_independent = maximum(element_nodes_p4est)

    # Replace the hanging slots: the hanging corner gets the midpoint of the coarse face and
    # the face midpoint gets a new hanging node, constrained by the nodes of the coarse face
    element_nodes = copy(element_nodes_p4est)
    hanging_masters = Vector{NTuple{3, Int}}()
    hanging_face = Vector{Int}(undef, 4)
    for element in 1:n_elements
        lnodes_decode_2d!(hanging_face, face_codes[element]) || continue
        for face in 1:4
            half = hanging_face[face]
            half < 0 && continue
            c1, c2 = P4EST_FACE_CORNERS_2D[face]
            regular_corner, hanging_corner = half == 0 ? (c1, c2) : (c2, c1)
            slot_a = Q2_CORNER_SLOTS[regular_corner + 1] + 1
            slot_b = Q2_CORNER_SLOTS[hanging_corner + 1] + 1
            slot_m = Q2_FACE_MIDPOINT_SLOTS[face] + 1

            # The slots refer to the nodes of the coarse face in the layout of the
            # coarse element
            a = element_nodes_p4est[slot_a, element]
            b = element_nodes_p4est[slot_b, element]
            m = element_nodes_p4est[slot_m, element]

            push!(hanging_masters, (a, m, b))
            element_nodes[slot_b, element] = m
            element_nodes[slot_m, element] = n_independent + length(hanging_masters)
        end
    end
    n_hanging = length(hanging_masters)
    hanging_masters_matrix = Matrix{Int}(undef, 3, n_hanging)
    hanging_coefficients = Matrix{Float64}(undef, 3, n_hanging)
    for k in 1:n_hanging
        hanging_masters_matrix[:, k] .= hanging_masters[k]
        hanging_coefficients[:, k] .= Q2_HANGING_COEFFICIENTS
    end

    coordinates = lnodes_q2_coordinates(mesh, element_nodes, n_independent + n_hanging)

    boundary_faces = lnodes_boundary_faces(mesh)
    boundary_entities, node_entity = lnodes_boundary_entities(boundary_faces,
                                                              element_nodes,
                                                              n_independent + n_hanging)

    return P4estLnodesQ2(element_nodes, coordinates, n_independent, n_hanging,
                         hanging_masters_matrix, hanging_coefficients, boundary_faces,
                         boundary_entities, node_entity)
end

"""
    apply_hanging_constraints!(values, lnodes::P4estLnodesQ2)

Set the values of the hanging nodes from the values of the independent nodes. `values` is a
vector of length `n_independent + n_hanging` or a matrix with this number of columns.
"""
function apply_hanging_constraints!(values::AbstractVector, lnodes::P4estLnodesQ2)
    @unpack n_independent, n_hanging, hanging_masters, hanging_coefficients = lnodes
    for k in 1:n_hanging
        values[n_independent + k] = sum(hanging_coefficients[i, k] *
                                        values[hanging_masters[i, k]] for i in 1:3)
    end
    return values
end

function apply_hanging_constraints!(values::AbstractMatrix, lnodes::P4estLnodesQ2)
    for row in axes(values, 1)
        apply_hanging_constraints!(view(values, row, :), lnodes)
    end
    return values
end

# Create the `p4est_lnodes` of degree 2, copy the element nodes (one-based) and face codes,
# and free the `p4est` structures again
function lnodes_q2_from_p4est(mesh)
    ghost = p4est_ghost_new(mesh.p4est, P4EST_CONNECT_FULL)
    lnodes = p4est_lnodes_new(mesh.p4est, ghost, 2)
    try
        lnodes_obj = unsafe_load(lnodes)
        @assert lnodes_obj.owned_count==lnodes_obj.num_local_nodes "serial run expected"
        n_elements = Int(lnodes_obj.num_local_elements)
        element_nodes = unsafe_wrap(Array, lnodes_obj.element_nodes,
                                    (Int(lnodes_obj.vnodes), n_elements))
        face_codes = unsafe_wrap(Array, lnodes_obj.face_code, n_elements)
        return Matrix{Int}(element_nodes) .+ 1, Vector{Int}(face_codes)
    finally
        p4est_lnodes_destroy(lnodes)
        p4est_ghost_destroy(ghost)
    end
end

# Port of `p4est_lnodes_decode` (2D). Set `hanging_face[face]` to the position of the
# element within the hanging face (0 or 1) or to -1 if the face is not hanging.
# Return whether any face is hanging.
function lnodes_decode_2d!(hanging_face, face_code)
    face_code == 0 && return false
    corner_faces = ((0, 2), (1, 2), (0, 3), (1, 3))
    corner_face_corners = ((0, -1, 0, -1), (-1, 0, 1, -1), (1, -1, -1, 0),
                           (-1, 1, -1, 1))
    fill!(hanging_face, -1)
    corner = face_code & 0x03
    work = face_code >> 2
    for i in 1:2
        face = corner_faces[corner + 1][i]
        if work & 0x01 != 0
            hanging_face[face + 1] = corner_face_corners[corner + 1][face + 1]
        end
        work >>= 1
    end
    return true
end

function lnodes_q2_coordinates(mesh, element_nodes, n_nodes)
    n_elements = size(element_nodes, 2)
    element_coordinates = Array{Float64}(undef, 2, 3, 3, n_elements)
    # Static nodes avoid allocations in `calc_node_coordinates!`
    calc_node_coordinates!(element_coordinates, mesh, SVector(-1.0, 0.0, 1.0))
    coordinates = Matrix{Float64}(undef, 2, n_nodes)
    for element in 1:n_elements, j in 1:3, i in 1:3
        node = element_nodes[i + 3 * (j - 1), element]
        coordinates[1, node] = element_coordinates[1, i, j, element]
        coordinates[2, node] = element_coordinates[2, i, j, element]
    end
    return coordinates
end

# Faces of the local quadrants that lie on a boundary face of their tree,
# with the boundary name of that tree face
function lnodes_boundary_faces(mesh)
    connectivity = mesh.p4est.connectivity
    n_trees = Int(connectivity.num_trees[])
    tree_to_tree = unsafe_wrap(Array, pointer(connectivity.tree_to_tree), (4, n_trees))
    tree_to_face = unsafe_wrap(Array, pointer(connectivity.tree_to_face), (4, n_trees))
    root_length = 2^P4EST_MAXLEVEL

    boundary_faces = Tuple{Int, Int, Symbol}[]
    for tree in 1:n_trees
        tree_pw = load_pointerwrapper_tree(mesh.p4est, tree)
        offset = Int(tree_pw.quadrants_offset[])
        quadrants = unsafe_wrap_sc(p4est_quadrant_t, tree_pw.quadrants)
        for (i, quadrant) in enumerate(quadrants)
            quadrant_length = 2^(P4EST_MAXLEVEL - quadrant.level)
            on_tree_face = (quadrant.x == 0,
                            quadrant.x + quadrant_length == root_length,
                            quadrant.y == 0,
                            quadrant.y + quadrant_length == root_length)
            for face in 1:4
                on_tree_face[face] || continue
                # A tree face without neighbor points to itself
                if tree_to_tree[face, tree] == tree - 1 &&
                   tree_to_face[face, tree] == face - 1
                    push!(boundary_faces,
                          (offset + i, face, mesh.boundary_names[face, tree]))
                end
            end
        end
    end
    return boundary_faces
end

# Collect the boundary names of each node and number the distinct sets of names.
# The names of a node are stored as a bit mask over the sorted list of all boundary names.
function lnodes_boundary_entities(boundary_faces, element_nodes, n_nodes)
    names = sort(unique(name for (_, _, name) in boundary_faces))
    @assert length(names)<=64 "at most 64 different boundary names are supported"
    node_mask = zeros(UInt64, n_nodes)
    for (element, face, name) in boundary_faces
        bit = one(UInt64) << (searchsortedfirst(names, name) - 1)
        for slot in Q2_FACE_SLOTS[face]
            node = element_nodes[slot + 1, element]
            node_mask[node] |= bit
        end
    end

    boundary_entities = Vector{Symbol}[]
    entity_ids = Dict{UInt64, Int}()
    node_entity = zeros(Int, n_nodes)
    for node in 1:n_nodes
        mask = node_mask[node]
        iszero(mask) && continue
        node_entity[node] = get!(entity_ids, mask) do
            push!(boundary_entities,
                  [names[i]
                   for i in eachindex(names)
                   if !iszero(mask & (one(UInt64) << (i - 1)))])
            return length(boundary_entities)
        end
    end
    return boundary_entities, node_entity
end

"""
    gridap_model(lnodes::P4estLnodesQ2)

Create a Gridap.jl `UnstructuredDiscreteModel` of the quadrilaterals of `lnodes`. The face
labeling contains the tags `"interior"`, `"boundary"`, and one tag for each boundary name of
the mesh. Faces and vertices are labeled with the boundary entities of `lnodes`, so that a
vertex at the junction of two boundaries belongs to both tags. Faces at hanging nodes are
interior faces.

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function gridap_model end

"""
    gridap_q2_space(lnodes::P4estLnodesQ2, model; dirichlet_tags = Symbol[])

Create a continuous Q2 finite element space on the Gridap.jl `model` created by
[`gridap_model`](@ref), using the node numbering of `lnodes`. Nodes carrying one of the
boundary names in `dirichlet_tags` are Dirichlet nodes. Hanging nodes are constrained with
`FESpaceWithLinearConstraints`.

Return a named tuple with
- `space`: the constrained space (the unconstrained one if there are no hanging nodes),
- `unconstrained`: the space with all nodes as degrees of freedom,
- `node_to_dof`: degree of freedom of each node in `unconstrained`, positive for free and
  negative for Dirichlet degrees of freedom.

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function gridap_q2_space end
"""
    gridap_q2_function(node_values, q2_space; field = 1)

Create a Gridap.jl `FEFunction` with the values `node_values` at all nodes of the
`P4estLnodesQ2` used to create `q2_space` with [`gridap_q2_space`](@ref), e.g., data
transferred with [`transfer_dg_to_q2!`](@ref). `node_values` is a vector or a matrix
`n_fields × n_nodes`, of which the row `field` is used. The function lives in the space
without hanging-node constraints, i.e., the values at hanging nodes are used as given;
with [`transfer_dg_to_q2!`](@ref), they satisfy the constraints.

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function gridap_q2_function end

"""
    gridap_node_values(uh, q2_space)

Values of the Gridap.jl function `uh` at all nodes of the `P4estLnodesQ2` used to create
`q2_space` with [`gridap_q2_space`](@ref), including the hanging nodes. The result can be
written into a DG solution with [`transfer_q2_to_dg!`](@ref).

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function gridap_node_values end
end # @muladd
