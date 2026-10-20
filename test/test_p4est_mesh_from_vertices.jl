module TestP4estMeshFromVertices

using Test
using Random: MersenneTwister
using Trixi
using Trixi: P4est
using AeroTrixi

# Mark cells with `flags[cell] > 0` and refine them (followed by 2:1 balancing),
# using the same mechanism as Trixi's `AMRCallback`
function refine_cells!(mesh, flags::Vector{Int})
    @assert length(flags) == Trixi.ncells(mesh)
    iter_volume_c = Trixi.cfunction(Trixi.copy_to_quad_iter_volume, Val(2))
    Trixi.iterate_p4est(mesh.p4est, flags; iter_volume_c)
    Trixi.refine!(mesh)
    return mesh
end

function cell_centers(mesh)
    coordinates = Array{Float64}(undef, 2, 2, 2, Trixi.ncells(mesh))
    Trixi.calc_node_coordinates!(coordinates, mesh, [-1.0, 1.0])
    return [sum(coordinates[:, i, j, cell] for i in 1:2, j in 1:2) / 4
            for cell in 1:Trixi.ncells(mesh)]
end

# Refine all cells whose center satisfies `predicate`, `levels` times
function refine_where!(mesh, predicate; levels = 1)
    for _ in 1:levels
        flags = [predicate(x) ? 1 : 0 for x in cell_centers(mesh)]
        refine_cells!(mesh, flags)
    end
    return mesh
end

function distance_to_segment(x, a, b)
    t = clamp(sum((x - a) .* (b - a)) / sum(abs2, b - a), 0, 1)
    return sqrt(sum(abs2, x - (a + t * (b - a))))
end

# Refinement patterns, each targeting one aspect of the mesh
function refine_pattern!(mesh, pattern, l)
    if pattern == :A # uniform
        refine_where!(mesh, x -> true)
    elseif pattern == :B # finer center block, hanging faces on all tree interfaces around it
        refine_where!(mesh, x -> max(abs(x[1]), abs(x[2])) < l)
    elseif pattern == :C # around a vertex shared by three trees
        refine_where!(mesh, x -> sqrt(sum(abs2, x .- l)) < 0.3; levels = 2)
    elseif pattern == :D # along the interface between two trapezoids
        refine_where!(mesh, x -> distance_to_segment(x, [l, -l], [1.0, -1.0]) < 0.3;
                      levels = 2)
    elseif pattern == :E # at the outer boundary of one trapezoid
        refine_where!(mesh, x -> x[1] > 0.8 && abs(x[2]) < 0.5; levels = 2)
    elseif pattern == :F # random
        rng = MersenneTwister(42)
        for _ in 1:3
            refine_cells!(mesh, [rand(rng) < 0.2 ? 1 : 0 for _ in 1:Trixi.ncells(mesh)])
        end
    end
    return mesh
end

# Port of `p4est_lnodes_decode` (2D): position of the element within each hanging face,
# -1 if the face is not hanging
function lnodes_decode(face_code)
    hanging_face = fill(-1, 4)
    face_code == 0 && return hanging_face
    corner_faces = ((0, 2), (1, 2), (0, 3), (1, 3))
    corner_face_corners = ((0, -1, 0, -1), (-1, 0, 1, -1), (1, -1, -1, 0), (-1, 1, -1, 1))
    c = face_code & 0x03
    work = face_code >> 2
    for i in 1:2
        f = corner_faces[c + 1][i]
        if work & 0x01 != 0
            hanging_face[f + 1] = corner_face_corners[c + 1][f + 1]
        end
        work >>= 1
    end
    return hanging_face
end

const FACE_CORNERS = ((0, 2), (1, 3), (0, 1), (2, 3))

# Node slot (zero-based, lexicographic) of a corner or face midpoint for degree 1 or 2
corner_slot(c, degree) = degree == 1 ? c : (0, 2, 6, 8)[c + 1]
face_midpoint_slot(f) = (3, 5, 1, 7)[f + 1]

"""
Check the `p4est_lnodes` numbering of degree 1 or 2 against the geometry of `mesh`:
- every independent node id has a unique position and all slots referring to it agree,
- no two node ids share a position (no duplicated nodes at tree corners),
- for each hanging face, the slots of the small element refer to the nodes of the coarse face
  in the coarse element's layout, i.e., the geometric hanging corner is the midpoint of
  the coarse face. This is the convention used to build hanging-node constraints.

Return the number of hanging faces.
"""
function check_lnodes(mesh, degree; atol = 1e-12)
    ghost = P4est.p4est_ghost_new(mesh.p4est, P4est.P4EST_CONNECT_FULL)
    lnodes = P4est.p4est_lnodes_new(mesh.p4est, ghost, degree)
    n_hanging_faces = 0
    try
        ln = unsafe_load(lnodes)
        n_elements = Int(ln.num_local_elements)
        n_nodes = Int(ln.num_local_nodes)
        @test n_elements == Trixi.ncells(mesh)
        @test ln.owned_count == n_nodes # serial run
        element_nodes = unsafe_wrap(Array, ln.element_nodes, (Int(ln.vnodes), n_elements))
        face_codes = unsafe_wrap(Array, ln.face_code, n_elements)

        n = degree + 1
        coordinates = Array{Float64}(undef, 2, n, n, n_elements)
        Trixi.calc_node_coordinates!(coordinates, mesh, collect(range(-1, 1; length = n)))
        position(element, slot) = coordinates[:, slot % n + 1, slot ÷ n + 1, element]
        node_id(element, slot) = Int(element_nodes[slot + 1, element]) + 1

        node_coordinates = fill(NaN, 2, n_nodes)
        max_mismatch = 0.0
        function assign!(id, x)
            if isnan(node_coordinates[1, id])
                node_coordinates[:, id] .= x
            else
                max_mismatch = max(max_mismatch, maximum(abs, node_coordinates[:, id] - x))
            end
        end

        # Assign positions from all slots that are not hanging
        for element in 1:n_elements
            hanging_face = lnodes_decode(face_codes[element])
            hanging_slots = Int[]
            for f in 0:3
                half = hanging_face[f + 1]
                half < 0 && continue
                # The corner of the face that is not shared with the coarse face is hanging
                push!(hanging_slots, corner_slot(FACE_CORNERS[f + 1][2 - half], degree))
                degree == 2 && push!(hanging_slots, face_midpoint_slot(f))
            end
            for slot in 0:(n^2 - 1)
                slot in hanging_slots && continue
                assign!(node_id(element, slot), position(element, slot))
            end
        end

        # Check the coarse-face convention of hanging faces
        for element in 1:n_elements
            hanging_face = lnodes_decode(face_codes[element])
            for f in 0:3
                half = hanging_face[f + 1]
                half < 0 && continue
                n_hanging_faces += 1
                regular_corner = corner_slot(FACE_CORNERS[f + 1][half + 1], degree)
                hanging_corner = corner_slot(FACE_CORNERS[f + 1][2 - half], degree)
                x_a = node_coordinates[:, node_id(element, regular_corner)]
                x_b = node_coordinates[:, node_id(element, hanging_corner)]
                x_hanging = position(element, hanging_corner)
                @test isapprox(x_a, position(element, regular_corner); atol)
                @test isapprox(x_hanging, (x_a + x_b) / 2; atol)
                if degree == 2
                    # The midpoint of the coarse face is an independent node for degree 2
                    x_m = node_coordinates[:, node_id(element, face_midpoint_slot(f))]
                    @test isapprox(x_m, (x_a + x_b) / 2; atol)
                end
            end
        end

        @test !any(isnan, node_coordinates)
        @test max_mismatch < atol
        rounded = Set(Tuple(round.(node_coordinates[:, id]; digits = 10))
                      for id in 1:n_nodes)
        @test length(rounded) == n_nodes
    finally
        P4est.p4est_lnodes_destroy(lnodes)
        P4est.p4est_ghost_destroy(ghost)
    end
    return n_hanging_faces
end

function tree_to_face(mesh)
    connectivity = mesh.p4est.connectivity
    n_trees = Int(connectivity.num_trees[])
    return [Int(connectivity.tree_to_face[i]) for i in 1:(4 * n_trees)]
end

# The residual of a constant state must vanish on any valid mesh, including mortars
function free_stream_residual(mesh)
    equations = CompressibleEulerEquations2D(1.4)
    initial_condition = (x, t, equations) -> prim2cons(SVector(1.0, 0.3, -0.2, 1.0),
                                                       equations)
    boundary_condition = BoundaryConditionDirichlet(initial_condition)
    boundary_conditions = (; x_neg = boundary_condition, x_pos = boundary_condition,
                           y_neg = boundary_condition, y_pos = boundary_condition)
    solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions)
    ode = semidiscretize(semi, (0.0, 1.0))
    du = similar(ode.u0)
    ode.f(du, ode.u0, semi, 0.0)
    return maximum(abs, du)
end

@testset "p4est_mesh_from_vertices" begin
    @testset "single tree" begin
        vertices = [0.0 2.0 0.0 1.0; 0.0 0.0 1.0 1.0]
        mesh = p4est_mesh_from_vertices(vertices, reshape(1:4, 4, 1); polydeg = 1)
        @test Trixi.ncells(mesh) == 1
        @test sort(vec(mesh.boundary_names)) == fill(:boundary, 4)
        @test mesh.tree_node_coordinates[:, 2, 1, 1] ≈ [2.0, 0.0]
        @test mesh.tree_node_coordinates[:, 2, 2, 1] ≈ [1.0, 1.0]

        # Left-handed trees are rejected
        @test_throws AssertionError p4est_mesh_from_vertices(vertices,
                                                             reshape([2, 1, 4, 3], 4, 1);
                                                             polydeg = 1)
    end

    @testset "3 × 3 Cartesian trees" begin
        # Each interior vertex is shared by four trees, two of which only touch at the corner.
        # Without corner connectivity, `p4est_lnodes` numbers these vertices more than once.
        xs = (-1.0, -1 / 3, 1 / 3, 1.0)
        vertices = reduce(hcat, [[x, y] for y in xs for x in xs])
        tree_to_vertex = reduce(hcat,
                                [[i + 4 * j, i + 1 + 4 * j, i + 4 + 4 * j, i + 5 + 4 * j]
                                 for j in 0:2 for i in 1:3])
        mesh = p4est_mesh_from_vertices(vertices, tree_to_vertex; polydeg = 1)
        @test mesh.p4est.connectivity.num_corners[] == 4
        @test check_lnodes(mesh, 1) == 0
        @test check_lnodes(mesh, 2) == 0
    end

    @testset "five trees: topology (trees_per_block = $n)" for n in (1, 2)
        mesh = p4est_mesh_five_trees(; polydeg = 2, trees_per_block = n)
        connectivity = mesh.p4est.connectivity
        @test connectivity.num_trees[] == 5 * n^2
        @test connectivity.num_vertices[] == (n == 1 ? 8 : 25)
        # Tree interfaces with flipped orientation (`tree_to_face >= 4` in 2D)
        @test any(>=(4), tree_to_face(mesh))
        # Boundary faces are named by the side of the box they lie on
        for name in (:x_neg, :x_pos, :y_neg, :y_pos)
            @test count(==(name), mesh.boundary_names) == n
        end
        @test count(==(Symbol("---")), mesh.boundary_names) == 4 * 5 * n^2 - 4 * n
        if n > 1
            # Vertices where diagonally opposite trees only touch at a corner
            @test connectivity.num_corners[] > 0
        end
    end

    @testset "five trees: lnodes (l = $l, trees_per_block = $n, pattern $pattern)" for l in (0.5,
                                                                                             0.4),
                                                                                       n in (1,
                                                                                             2),
                                                                                       pattern in (:A,
                                                                                                   :B,
                                                                                                   :C,
                                                                                                   :D,
                                                                                                   :E,
                                                                                                   :F)

        mesh = p4est_mesh_five_trees(; l, polydeg = 2, trees_per_block = n,
                                     initial_refinement_level = 1)
        refine_pattern!(mesh, pattern, l)
        n_hanging_q1 = check_lnodes(mesh, 1)
        n_hanging_q2 = check_lnodes(mesh, 2)
        @test n_hanging_q1 == n_hanging_q2
        if pattern == :A
            @test n_hanging_q1 == 0
        else
            @test n_hanging_q1 > 0
        end
    end

    @testset "five trees: free-stream preservation (pattern $pattern)" for pattern in (:A,
                                                                                       :B,
                                                                                       :C,
                                                                                       :F)
        mesh = p4est_mesh_five_trees(; l = 0.4, polydeg = 3, trees_per_block = 2,
                                     initial_refinement_level = 1)
        refine_pattern!(mesh, pattern, 0.4)
        # Round-off grows with the inverse cell size; a broken interface gives O(1e-3) or more
        @test free_stream_residual(mesh) < 1e-9
    end
end

end # module
