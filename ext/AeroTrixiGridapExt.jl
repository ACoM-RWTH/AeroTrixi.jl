module AeroTrixiGridapExt

using AeroTrixi: AeroTrixi, P4estLnodesQ2
using Gridap: Gridap
using Gridap.Arrays: Table, Fill
using Gridap.Fields: Point
using Gridap.ReferenceFEs: QUAD, LagrangianRefFE
using Gridap.Geometry: UnstructuredGrid, UnstructuredGridTopology,
                       UnstructuredDiscreteModel, FaceLabeling, Triangulation, get_faces,
                       num_faces
using Gridap.FESpaces: FESpaceWithLinearConstraints, UnconstrainedFESpace, CellFE,
                       H1Conformity, compute_cell_space, FEFunction,
                       interpolate_everywhere, get_free_dof_values,
                       get_dirichlet_dof_values,
                       num_free_dofs, num_dirichlet_dofs

# Node slots (one-based, lexicographic as in Trixi.jl) of the local nodes of a Gridap Q2
# quadrilateral: four vertices, the midpoints of the edges (bottom, top, left, right), and
# the center
const GRIDAP_Q2_TO_LEXICOGRAPHIC = (1, 3, 7, 9, 2, 8, 4, 6, 5)

# Local Gridap edge of each `p4est` face (left, right, bottom, top)
const P4EST_FACE_TO_GRIDAP_EDGE = (3, 4, 1, 2)

# Entity 1 is the interior, entity `k + 1` is `lnodes.boundary_entities[k]`
node_entity(lnodes, node) = lnodes.node_entity[node] + 1

function AeroTrixi.gridap_model(lnodes::P4estLnodesQ2)
    element_nodes, coordinates = lnodes.element_nodes, lnodes.coordinates
    n_elements = size(element_nodes, 2)

    # Vertices of the Gridap grid are the corner nodes of the elements
    node_to_vertex = zeros(Int32, size(coordinates, 2))
    vertex_to_node = Int[]
    cell_vertices = Vector{Int32}(undef, 4 * n_elements)
    for element in 1:n_elements, i in 1:4
        node = element_nodes[GRIDAP_Q2_TO_LEXICOGRAPHIC[i], element]
        if node_to_vertex[node] == 0
            push!(vertex_to_node, node)
            node_to_vertex[node] = length(vertex_to_node)
        end
        cell_vertices[4 * (element - 1) + i] = node_to_vertex[node]
    end
    vertex_coordinates = [Point(coordinates[1, node], coordinates[2, node])
                          for node in vertex_to_node]
    cell_vertex_ids = Table(cell_vertices, Int32.(1:4:(4 * n_elements + 1)))
    grid = UnstructuredGrid(vertex_coordinates, cell_vertex_ids,
                            [LagrangianRefFE(Float64, QUAD, 1)], ones(Int8, n_elements))
    topology = UnstructuredGridTopology(grid)

    labels = face_labeling(lnodes, topology, vertex_to_node)

    return UnstructuredDiscreteModel(grid, topology, labels)
end

# Labels from the boundary faces and entities of `lnodes`. Gridap's default labeling cannot
# be used, since it would mark the faces at hanging nodes as boundary faces.
function face_labeling(lnodes, topology, vertex_to_node)
    vertex_entity = Int32[node_entity(lnodes, node) for node in vertex_to_node]

    edge_entity = ones(Int32, num_faces(topology, 1))
    cell_edges = get_faces(topology, 2, 1)
    for (element, face, _) in lnodes.boundary_faces
        edge = cell_edges[element][P4EST_FACE_TO_GRIDAP_EDGE[face]]
        # The midpoint of a boundary face only carries the name of this face
        midpoint = lnodes.element_nodes[(4, 6, 2, 8)[face], element]
        edge_entity[edge] = node_entity(lnodes, midpoint)
    end

    cell_entity = ones(Int32, num_faces(topology, 2))

    entities = lnodes.boundary_entities
    names = sort(unique(reduce(vcat, entities; init = Symbol[])))
    tag_to_name = ["interior"; "boundary"; String.(names)]
    tag_to_entities = [Int32[1], Int32.(2:(length(entities) + 1))]
    for name in names
        push!(tag_to_entities,
              Int32[k + 1 for k in eachindex(entities)
                    if name in entities[k]])
    end

    return FaceLabeling([vertex_entity, edge_entity, cell_entity], tag_to_entities,
                        tag_to_name)
end

function AeroTrixi.gridap_q2_space(lnodes::P4estLnodesQ2, model;
                                   dirichlet_tags = Symbol[])
    element_nodes = lnodes.element_nodes
    n_independent, n_hanging = lnodes.n_independent, lnodes.n_hanging
    n_elements = size(element_nodes, 2)
    n = n_independent + n_hanging
    tags = Symbol.(collect(dirichlet_tags))

    # Free and Dirichlet numbering of the nodes. A node is a Dirichlet node if one of its
    # boundary names is a Dirichlet tag; the first matching tag is used.
    node_to_dof = Vector{Int}(undef, n)
    dirichlet_dof_tag = Int8[]
    n_free = 0
    for node in 1:n
        entity = lnodes.node_entity[node]
        tag = entity == 0 ? nothing :
              findfirst(in(lnodes.boundary_entities[entity]), tags)
        if tag === nothing
            n_free += 1
            node_to_dof[node] = n_free
        else
            push!(dirichlet_dof_tag, tag)
            node_to_dof[node] = -length(dirichlet_dof_tag)
        end
    end
    n_dirichlet = length(dirichlet_dof_tag)

    cell_dofs = Vector{Int32}(undef, 9 * n_elements)
    cell_is_dirichlet = falses(n_elements)
    for element in 1:n_elements, i in 1:9
        dof = node_to_dof[element_nodes[GRIDAP_Q2_TO_LEXICOGRAPHIC[i], element]]
        cell_dofs[9 * (element - 1) + i] = dof
        dof < 0 && (cell_is_dirichlet[element] = true)
    end
    cell_dof_ids = Table(cell_dofs, Int32.(1:9:(9 * n_elements + 1)))

    # Shape functions and DOF basis of the Q2 elements
    cell_fe = CellFE(model, Fill(LagrangianRefFE(Float64, QUAD, 2), n_elements),
                     H1Conformity())
    fe_basis, fe_dof_basis = compute_cell_space(cell_fe, Triangulation(model))
    unconstrained = UnconstrainedFESpace(Vector{Float64}, n_free, n_dirichlet,
                                         cell_dof_ids, fe_basis, fe_dof_basis,
                                         collect(cell_is_dirichlet), dirichlet_dof_tag,
                                         Int32.(findall(cell_is_dirichlet)),
                                         length(tags))

    if n_hanging == 0
        return (; space = unconstrained, unconstrained, node_to_dof)
    end

    hanging_masters, hanging_coefficients = lnodes.hanging_masters,
                                            lnodes.hanging_coefficients
    slave_dofs = node_to_dof[(n_independent + 1):n]
    master_dofs = Table(vec(node_to_dof[hanging_masters]), 1:3:(3 * n_hanging + 1))
    coefficients = Table(vec(hanging_coefficients), 1:3:(3 * n_hanging + 1))
    space = FESpaceWithLinearConstraints(slave_dofs, master_dofs, coefficients,
                                         unconstrained)

    return (; space, unconstrained, node_to_dof)
end

function AeroTrixi.gridap_q2_function(node_values, q2_space)
    @assert length(node_values)==length(q2_space.node_to_dof) "node_values must contain all nodes"
    space = q2_space.unconstrained
    free_values = zeros(eltype(node_values), num_free_dofs(space))
    dirichlet_values = zeros(eltype(node_values), num_dirichlet_dofs(space))
    for (node, dof) in enumerate(q2_space.node_to_dof)
        if dof > 0
            free_values[dof] = node_values[node]
        else
            dirichlet_values[-dof] = node_values[node]
        end
    end
    return FEFunction(space, free_values, dirichlet_values)
end

function AeroTrixi.gridap_node_values(uh, q2_space)
    # Interpolation onto the space without constraints evaluates the function at all
    # nodes, including the hanging nodes. `interpolate_everywhere` also sets the values of
    # the Dirichlet nodes, which `interpolate` would set to zero.
    u_nodes = interpolate_everywhere(uh, q2_space.unconstrained)
    free_values = get_free_dof_values(u_nodes)
    dirichlet_values = get_dirichlet_dof_values(u_nodes)
    return [dof > 0 ? free_values[dof] : dirichlet_values[-dof]
            for dof in q2_space.node_to_dof]
end

end # module
