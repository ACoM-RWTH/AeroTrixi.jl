module AeroTrixiGridapExt

using AeroTrixi: AeroTrixi, P4estLnodesQ2, GridapLinearProblem, gridap_setup,
                 gridap_solve, transfer_dg_to_q2!, transfer_q2_to_dg!,
                 gridap_model, gridap_q2_space, gridap_q2_function, gridap_node_values,
                 apply_hanging_constraints!,
                 DiscreteCallback, isfinished, derivative_discontinuity!,
                 mesh_equations_solver_cache, @trixi_timeit, timer, ncells, summary_box,
                 @unpack, wrap_array, get_node_vars, get_node_coords
using Gridap: Gridap
using Gridap.CellData: Measure
using Gridap.FESpaces: TrialFESpace, assemble_matrix, assemble_vector
using StaticArrays: SVector
using LinearAlgebra: mul!, norm
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
                       num_free_dofs, num_dirichlet_dofs, SingleFieldFEFunction

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
        return (; space = unconstrained, unconstrained, node_to_dof, lnodes)
    end

    hanging_masters, hanging_coefficients = lnodes.hanging_masters,
                                            lnodes.hanging_coefficients
    slave_dofs = node_to_dof[(n_independent + 1):n]
    # The hanging nodes are the last free degrees of freedom. Since the constrained space
    # numbers its free degrees of freedom by keeping the order of the remaining ones, the
    # free degree of freedom `dof` of an independent node is the same in both spaces, which
    # `gridap_node_values` uses.
    @assert slave_dofs == (n_free - n_hanging + 1):n_free
    master_dofs = Table(vec(node_to_dof[hanging_masters]), 1:3:(3 * n_hanging + 1))
    coefficients = Table(vec(hanging_coefficients), 1:3:(3 * n_hanging + 1))
    space = FESpaceWithLinearConstraints(slave_dofs, master_dofs, coefficients,
                                         unconstrained)

    return (; space, unconstrained, node_to_dof, lnodes)
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
    return gridap_node_values!(Vector{Float64}(undef, length(q2_space.node_to_dof)), uh,
                               q2_space)
end

# A function of the Q2 space (with or without constraints) is read from its degrees of
# freedom; the values at hanging nodes of the constrained space follow from the
# constraints. Other functions are interpolated.
function gridap_node_values!(values, uh::SingleFieldFEFunction, q2_space)
    free_values = get_free_dof_values(uh)
    dirichlet_values = get_dirichlet_dof_values(uh)
    n_dirichlet = num_dirichlet_dofs(q2_space.unconstrained)
    if length(dirichlet_values) != n_dirichlet
        return gridap_node_values_interpolated!(values, uh, q2_space)
    end
    if length(free_values) == num_free_dofs(q2_space.unconstrained)
        n_read = length(q2_space.node_to_dof)
    elseif length(free_values) == num_free_dofs(q2_space.space)
        n_read = q2_space.lnodes.n_independent
    else
        return gridap_node_values_interpolated!(values, uh, q2_space)
    end
    # Function barrier: the vectors are stored with abstract types in the `FEFunction`
    node_values_from_dofs!(values, free_values, dirichlet_values, q2_space.node_to_dof,
                           n_read)
    if n_read < length(values)
        apply_hanging_constraints!(values, q2_space.lnodes)
    end
    return values
end

function node_values_from_dofs!(values, free_values, dirichlet_values, node_to_dof, n_read)
    @inbounds for node in 1:n_read
        dof = node_to_dof[node]
        values[node] = dof > 0 ? free_values[dof] : dirichlet_values[-dof]
    end
    return values
end

function gridap_node_values!(values, uh, q2_space)
    gridap_node_values_interpolated!(values, uh,
                                     q2_space)
end

function gridap_node_values_interpolated!(values, uh, q2_space)
    # Interpolation onto the space without constraints evaluates the function at all
    # nodes, including the hanging nodes. `interpolate_everywhere` also sets the values of
    # the Dirichlet nodes, which `interpolate` would set to zero.
    u_nodes = interpolate_everywhere(uh, q2_space.unconstrained)
    free_values = get_free_dof_values(u_nodes)
    dirichlet_values = get_dirichlet_dof_values(u_nodes)
    for (node, dof) in enumerate(q2_space.node_to_dof)
        values[node] = dof > 0 ? free_values[dof] : dirichlet_values[-dof]
    end
    return values
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
    end
    @trixi_timeit timer() "solve" begin
        solution = gridap_solve(update.problem, setup, q2, source_nodes)
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
    # Number of source fields from the source at the first DG node
    u = wrap_array(u_ode, semi)
    u_node = get_node_vars(u, equations, solver, 1, 1, 1)
    x_node = get_node_coords(cache.elements.node_coordinates, equations, solver, 1, 1, 1)
    n_sources = length(update.source(u_node, x_node, equations))
    @trixi_timeit timer() "problem" begin
        setup = gridap_setup(update.problem, q2, model, n_sources)
    end
    source_nodes = zeros(n_sources, length(q2.node_to_dof))
    return GridapMeshData(Int(mesh.p4est.revision[]), ncells(mesh), lnodes, q2, setup,
                          source_nodes)
end

# Linear problems: assemble and factorize the matrix once per mesh
function AeroTrixi.gridap_setup(problem::GridapLinearProblem, q2, model, n_sources)
    # Homogeneous Dirichlet conditions
    trial_space = TrialFESpace(q2.space, 0.0)
    dΩ = Measure(Triangulation(model), problem.quadrature_degree)
    @trixi_timeit timer() "assemble" begin
        A = assemble_matrix((u, v) -> problem.bilinear_form(u, v, dΩ), trial_space,
                            q2.space)
    end
    @trixi_timeit timer() "factorize" factorization=problem.factorize(A)
    load = nothing
    if problem.precompute_load
        @trixi_timeit timer() "load matrices" begin
            load = load_matrices(problem, q2, model, dΩ, n_sources)
        end
    end
    return (; trial_space, dΩ, factorization, load)
end

# For a linear form that is affine in the sources, the load vector is
# `b0 + Σ_k B_k f_k` with the source values `f_k` at all nodes
function load_matrices(problem, q2, model, dΩ, n_sources)
    # Space with all nodes as free degrees of freedom, in the order of the nodes
    source_space = gridap_q2_space(q2.lnodes, model).unconstrained
    zero_source = zero(source_space)
    zero_sources = ntuple(_ -> zero_source, n_sources)
    message = "the linear form is not affine in the sources; use " *
              "`GridapLinearProblem(...; precompute_load = false)`"
    b0, B = try
        b0 = assemble_vector(v -> problem.linear_form(v, zero_sources, dΩ), q2.space)
        B = map(1:n_sources) do k
            assemble_matrix((f, v) -> problem.linear_form(v,
                                                          ntuple(j -> j == k ? f :
                                                                      zero_source,
                                                                 n_sources), dΩ),
                            source_space, q2.space)
        end
        b0, B
    catch err
        # E.g., a product of sources is a product of trial functions here, which Gridap.jl
        # rejects
        throw(ArgumentError(message * " (assembly failed with " *
                            sprint(showerror, err) * ")"))
    end
    b = similar(b0)

    # Check with random sources that the linear form is affine in the sources
    source_nodes = rand(n_sources, length(q2.node_to_dof))
    sources = ntuple(k -> gridap_q2_function(source_nodes, q2; field = k), n_sources)
    b_assembled = assemble_vector(v -> problem.linear_form(v, sources, dΩ), q2.space)
    load_vector!(b, (; b0, B), source_nodes)
    if norm(b - b_assembled) > 1e-10 * (1 + norm(b_assembled))
        throw(ArgumentError(message))
    end
    return (; b0, B, b)
end

function load_vector!(b, load, source_nodes)
    copyto!(b, load.b0)
    for k in eachindex(load.B)
        mul!(b, load.B[k], view(source_nodes, k, :), true, true)
    end
    return b
end

function AeroTrixi.gridap_solve(problem::GridapLinearProblem, setup, q2, source_nodes)
    @unpack trial_space, dΩ, factorization, load = setup
    if load === nothing
        sources = ntuple(k -> gridap_q2_function(source_nodes, q2; field = k),
                         size(source_nodes, 1))
        b = assemble_vector(v -> problem.linear_form(v, sources, dΩ), q2.space)
    else
        b = load_vector!(load.b, load, source_nodes)
    end
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
