module AeroTrixiGridapExt

using AeroTrixi: AeroTrixi, P4estLnodesQ2, GridapLinearProblem, gridap_setup,
                 gridap_solve, transfer_dg_to_q2!, transfer_q2_to_dg!,
                 gridap_model, gridap_q2_space, gridap_q2_function, gridap_node_values,
                 DiscreteCallback, isfinished, derivative_discontinuity!,
                 mesh_equations_solver_cache, @trixi_timeit, timer, ncells, summary_box,
                 @unpack, wrap_array, get_node_vars, get_node_coords
using Gridap: Gridap
using Gridap.CellData: Measure
using Gridap.FESpaces: TrialFESpace, assemble_matrix, assemble_vector
using StaticArrays: SVector
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

function AeroTrixi.gridap_q2_function(node_values::AbstractMatrix, q2_space; field = 1)
    return AeroTrixi.gridap_q2_function(view(node_values, field, :), q2_space)
end

function AeroTrixi.gridap_q2_function(node_values::AbstractVector, q2_space)
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

###############################################################################
# Callback solving a finite element problem with Gridap.jl and writing the result into the
# DG solution

# Mesh-dependent data, rebuilt when the mesh changes
struct GridapMeshData{Lnodes, Q2, Setup}
    revision     :: Int
    n_elements   :: Int
    lnodes       :: Lnodes
    q2           :: Q2
    setup        :: Setup
    source_nodes :: Matrix{Float64}
end

mutable struct GridapCallbackUpdate{Problem, Source, Write, Tags}
    const problem        :: Problem
    const source         :: Source
    const write          :: Write
    const dirichlet_tags :: Tags
    const interval       :: Int
    mesh_data            :: Any
    # Set by `gridap_stage_update`: the problem is solved on each Runge-Kutta stage, and the
    # callback only acts after mesh changes
    stages   :: Bool
    n_setups :: Int
    n_solves :: Int
end

function AeroTrixi.GridapCallback(semi, problem; source, write, dirichlet_tags = (),
                                  interval = 1)
    update = GridapCallbackUpdate(problem, source, write, Tuple(Symbol.(dirichlet_tags)),
                                  Int(interval), nothing, false, 0, 0)
    condition = function (u, t, integrator)
        if update.stages
            # The stage updates keep the solution up to date as long as the mesh is not
            # changed
            mesh, _, _, _ = mesh_equations_solver_cache(integrator.p)
            return mesh_changed(update.mesh_data, mesh)
        end
        return interval > 0 &&
               (integrator.stats.naccept % interval == 0 || isfinished(integrator))
    end
    return DiscreteCallback(condition, update, save_positions = (false, false),
                            initialize = initialize_gridap_callback!)
end

function AeroTrixi.gridap_stage_update(cb::DiscreteCallback{<:Any,
                                                            <:GridapCallbackUpdate})
    update = cb.affect!
    update.stages = true
    return update
end

# Compute the solution for the initial condition
function initialize_gridap_callback!(cb, u, t, integrator)
    cb.affect!(integrator)
    return nothing
end

function (update::GridapCallbackUpdate)(integrator)
    semi = integrator.p
    @trixi_timeit timer() "Gridap" begin
        update_gridap!(integrator.u, semi, update)
    end
    # The solution vector has been modified
    derivative_discontinuity!(integrator, true)
    return nothing
end

# Called as `stage_limiter!` by the time integration method on each stage value, before the
# right-hand side is evaluated, and on the solution at the end of each step
function (update::GridapCallbackUpdate)(u_ode, integrator, semi, t)
    @trixi_timeit timer() "Gridap (stages)" begin
        update_gridap!(u_ode, semi, update)
    end
    return nothing
end

function mesh_changed(data, mesh)
    return data === nothing || data.revision != mesh.p4est.revision[] ||
           data.n_elements != ncells(mesh)
end

# Solve the problem with the sources from `update.source` and write the result with
# `update.write`
function update_gridap!(u_ode, semi, update::GridapCallbackUpdate)
    mesh, equations, _, _ = mesh_equations_solver_cache(semi)
    data = update.mesh_data
    if mesh_changed(data, mesh)
        @trixi_timeit timer() "setup" begin
            data = gridap_mesh_data(u_ode, semi, update)
        end
        update.mesh_data = data
        update.n_setups += 1
    end
    @unpack lnodes, q2, setup, source_nodes = data

    @trixi_timeit timer() "DG → Q2" begin
        transfer_dg_to_q2!(update.source, source_nodes, u_ode, semi, lnodes)
        sources = ntuple(field -> gridap_q2_function(source_nodes, q2; field),
                         size(source_nodes, 1))
    end
    @trixi_timeit timer() "solve" begin
        solution = gridap_solve(update.problem, setup, q2, sources)
    end
    @trixi_timeit timer() "Q2 → DG" begin
        if solution isa Tuple
            # Solution fields as rows
            solution_nodes = permutedims(reduce(hcat,
                                                map(uh -> gridap_node_values(uh, q2),
                                                    solution)))
        else
            solution_nodes = gridap_node_values(solution, q2)
        end
        transfer_q2_to_dg!(update.write, u_ode, semi, lnodes, solution_nodes)
    end
    update.n_solves += 1
    return nothing
end

function gridap_mesh_data(u_ode, semi, update)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    @trixi_timeit timer() "lnodes" lnodes=P4estLnodesQ2(mesh)
    @trixi_timeit timer() "model" model=gridap_model(lnodes)
    @trixi_timeit timer() "space" begin
        q2 = gridap_q2_space(lnodes, model; dirichlet_tags = update.dirichlet_tags)
    end
    @trixi_timeit timer() "problem" setup=gridap_setup(update.problem, q2, model)
    # Number of source fields from the source at the first DG node
    u = wrap_array(u_ode, semi)
    u_node = get_node_vars(u, equations, solver, 1, 1, 1)
    x_node = get_node_coords(cache.elements.node_coordinates, equations, solver, 1, 1, 1)
    n_sources = length(update.source(u_node, x_node, equations))
    source_nodes = zeros(n_sources, length(q2.node_to_dof))
    return GridapMeshData(Int(mesh.p4est.revision[]), ncells(mesh), lnodes, q2, setup,
                          source_nodes)
end

# Linear problems: assemble and factorize the matrix once per mesh
function AeroTrixi.gridap_setup(problem::GridapLinearProblem, q2, model)
    # Homogeneous Dirichlet conditions
    trial_space = TrialFESpace(q2.space, 0.0)
    dΩ = Measure(Triangulation(model), problem.quadrature_degree)
    @trixi_timeit timer() "assemble" begin
        A = assemble_matrix((u, v) -> problem.bilinear_form(u, v, dΩ), trial_space,
                            q2.space)
    end
    @trixi_timeit timer() "factorize" factorization=problem.factorize(A)
    return (; trial_space, dΩ, factorization)
end

function AeroTrixi.gridap_solve(problem::GridapLinearProblem, setup, q2, sources)
    @unpack trial_space, dΩ, factorization = setup
    b = assemble_vector(v -> problem.linear_form(v, sources, dΩ), q2.space)
    return FEFunction(trial_space, factorization \ b)
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:GridapCallbackUpdate})
    update = cb.affect!
    if update.stages
        print(io, "GridapCallback(stages)")
    else
        print(io, "GridapCallback(interval=", update.interval, ")")
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:GridapCallbackUpdate})
    if get(io, :compact, false)
        show(io, cb)
    else
        update = cb.affect!
        setup = [
            "update" => update.stages ? "each Runge-Kutta stage" :
                        "every $(update.interval) time steps",
            "problem" => nameof(typeof(update.problem)),
            "Dirichlet tags" => join(update.dirichlet_tags, ", "),
            "source" => nameof(typeof(update.source)),
            "write" => nameof(typeof(update.write))]
        summary_box(io, "GridapCallback", setup)
    end
    return nothing
end

end # module
