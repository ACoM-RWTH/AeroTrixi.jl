# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing-EC-Trixi for further details.
@muladd begin
#! format: noindent

"""
    p4est_mesh_from_vertices(vertices, tree_to_vertex; polydeg,
                             boundary_name = Returns(:boundary),
                             RealT = Float64, initial_refinement_level = 0,
                             unsaved_changes = true,
                             p4est_partition_allow_for_coarsening = true)

Create a 2D `P4estMesh` from an unstructured list of straight-sided quadrilateral trees.

In contrast to `P4estMesh(trees_per_dimension; ...)`, the `p4est` connectivity is derived
from the vertex numbering with `p4est_connectivity_complete`. It therefore contains the full
face *and corner* connectivity between trees, including trees that only touch at a corner and
vertices shared by an arbitrary number of trees. A complete connectivity is required by
continuous node numberings such as `p4est_lnodes`.

# Arguments
- `vertices::AbstractMatrix`: `2 × n_vertices` matrix of vertex coordinates.
- `tree_to_vertex::AbstractMatrix{<:Integer}`: `4 × n_trees` matrix of one-based vertex indices.
  The vertices of each tree are ordered as in `p4est` (z-order), i.e., `(-1, -1)`, `(1, -1)`,
  `(-1, 1)`, `(1, 1)` in the reference coordinates of the tree. Each tree must be right-handed.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
- `boundary_name`: function mapping the midpoint `x::SVector{2}` of a boundary face to the
  `Symbol` used as boundary name. Default: `Returns(:boundary)`.
- `RealT::Type`: the type that should be used for coordinates. Default: `Float64`.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the
  simulation starts. Default: `0`.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file. Default: `true`.
- `p4est_partition_allow_for_coarsening::Bool`: Must be `true` when using AMR to make mesh
  adaptivity independent of domain partitioning. Default: `true`.

Periodicity is not supported, since it is not inherent in the list of vertices.
"""
function p4est_mesh_from_vertices(vertices::AbstractMatrix,
                                  tree_to_vertex::AbstractMatrix{<:Integer};
                                  polydeg,
                                  boundary_name = Returns(:boundary),
                                  RealT = Float64, initial_refinement_level = 0,
                                  unsaved_changes = true,
                                  p4est_partition_allow_for_coarsening = true)
    @assert size(vertices, 1)==2 "`vertices` must be a 2 × n_vertices matrix"
    @assert size(tree_to_vertex, 1)==4 "`tree_to_vertex` must be a 4 × n_trees matrix"
    @assert all(v -> 1 <= v <= size(vertices, 2), tree_to_vertex) "invalid vertex index in `tree_to_vertex`"

    n_trees = size(tree_to_vertex, 2)
    for tree in 1:n_trees
        @assert tree_jacobian_is_positive(vertices, tree_to_vertex, tree) "tree $tree is not right-handed"
    end

    connectivity = connectivity_from_vertices(vertices, tree_to_vertex)

    basis = LobattoLegendreBasis(RealT, polydeg)
    nodes = basis.nodes
    tree_node_coordinates = Array{RealT, 4}(undef, 2, length(nodes), length(nodes),
                                            n_trees)
    calc_tree_node_coordinates_bilinear!(tree_node_coordinates, nodes, vertices,
                                         tree_to_vertex)

    boundary_names = boundary_names_from_connectivity(connectivity, vertices,
                                                      tree_to_vertex, boundary_name)

    p4est = new_p4est(connectivity, initial_refinement_level)

    return P4estMesh{2}(p4est, tree_node_coordinates, nodes, boundary_names, "",
                        unsaved_changes, p4est_partition_allow_for_coarsening)
end

# Corners of each face in p4est numbering (zero-based faces and corners, as in `p4est_face_corners`)
const P4EST_FACE_CORNERS_2D = ((0, 2), (1, 3), (0, 1), (2, 3))

# Bilinear shape functions of the four tree corners in z-order
@inline function bilinear_shape_functions(xi, eta)
    return SVector((1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                   (1 - xi) * (1 + eta), (1 + xi) * (1 + eta)) / 4
end

function tree_jacobian_is_positive(vertices, tree_to_vertex, tree)
    v = ntuple(c -> SVector(vertices[1, tree_to_vertex[c, tree]],
                            vertices[2, tree_to_vertex[c, tree]]), 4)
    # The Jacobian determinant of a bilinear map is linear in each reference coordinate,
    # so it is positive everywhere if it is positive at all four corners.
    for (a, b, c) in ((1, 2, 3), (2, 4, 1), (3, 1, 4), (4, 3, 2))
        d1 = v[b] - v[a]
        d2 = v[c] - v[a]
        if d1[1] * d2[2] - d1[2] * d2[1] <= 0
            return false
        end
    end
    return true
end

function connectivity_from_vertices(vertices, tree_to_vertex)
    n_vertices = size(vertices, 2)
    n_trees = size(tree_to_vertex, 2)

    # p4est always stores three coordinates per vertex
    vertices_3d = zeros(Float64, 3, n_vertices)
    vertices_3d[1:2, :] .= vertices

    # Subtract 1 because `p4est` uses zero-based indexing
    tree_to_vertex_p4est = Matrix{p4est_topidx_t}(tree_to_vertex .- 1)

    # Trees initially point to themselves, which is a valid connectivity;
    # `p4est_connectivity_complete` overwrites these fields
    tree_to_tree = Matrix{p4est_topidx_t}(undef, 4, n_trees)
    tree_to_face = Matrix{Int8}(undef, 4, n_trees)
    for tree in 1:n_trees, face in 1:4
        tree_to_tree[face, tree] = tree - 1
        tree_to_face[face, tree] = face - 1
    end

    # No corner connectivity yet; it is derived by `p4est_connectivity_complete`
    n_corners = 0
    tree_to_corner = C_NULL
    ctt_offset = zeros(p4est_topidx_t, 1)
    corner_to_tree = C_NULL
    corner_to_corner = C_NULL

    connectivity = p4est_connectivity_new_copy(n_vertices, n_trees, n_corners,
                                               vertices_3d, tree_to_vertex_p4est,
                                               tree_to_tree, tree_to_face,
                                               tree_to_corner, ctt_offset,
                                               corner_to_tree, corner_to_corner)

    p4est_connectivity_complete(connectivity)

    @assert p4est_connectivity_is_valid(connectivity) == 1

    return connectivity
end

function calc_tree_node_coordinates_bilinear!(tree_node_coordinates, nodes, vertices,
                                              tree_to_vertex)
    for tree in axes(tree_node_coordinates, 4)
        for j in eachindex(nodes), i in eachindex(nodes)
            shape = bilinear_shape_functions(nodes[i], nodes[j])
            for dim in 1:2
                tree_node_coordinates[dim, i, j, tree] = sum(shape[c] *
                                                             vertices[dim,
                                                                      tree_to_vertex[c,
                                                                                     tree]]
                                                             for c in 1:4)
            end
        end
    end

    return tree_node_coordinates
end

function boundary_names_from_connectivity(connectivity, vertices, tree_to_vertex,
                                          boundary_name)
    n_trees = size(tree_to_vertex, 2)
    conn = unsafe_load(connectivity)
    tree_to_tree = unsafe_wrap(Array, conn.tree_to_tree, (4, n_trees))
    tree_to_face = unsafe_wrap(Array, conn.tree_to_face, (4, n_trees))

    # Interior faces keep the default name "---" used by Trixi.jl
    boundary_names = fill(Symbol("---"), 4, n_trees)
    for tree in 1:n_trees, face in 1:4
        # A face without neighbor points to itself
        if tree_to_tree[face, tree] == tree - 1 && tree_to_face[face, tree] == face - 1
            c1, c2 = P4EST_FACE_CORNERS_2D[face]
            v1 = tree_to_vertex[c1 + 1, tree]
            v2 = tree_to_vertex[c2 + 1, tree]
            midpoint = SVector((vertices[1, v1] + vertices[1, v2]) / 2,
                               (vertices[2, v1] + vertices[2, v2]) / 2)
            boundary_names[face, tree] = boundary_name(midpoint)
        end
    end

    return boundary_names
end

"""
    p4est_mesh_five_trees(; l = 0.5, trees_per_block = 1, polydeg, kwargs...)

Create a 2D `P4estMesh` of the box `[-1, 1]^2` consisting of five blocks: the square
`[-l, l]^2` in the middle and four trapezoids connecting its sides to the sides of the box.
Each block is split into `trees_per_block × trees_per_block` trees. The connectivity is created
with [`p4est_mesh_from_vertices`](@ref) and contains the full face and corner connectivity.

The mesh is meant as a simple workhorse for testing features of unstructured meshes:
- trees are non-affine (bilinear) quadrilaterals,
- the trapezoids are rotated copies of each other, so that the face numbers on both sides of
  a tree interface differ and some interfaces have flipped orientation,
- the vertices `(±l, ±l)` are shared by three trees,
- for `trees_per_block > 1`, there are vertices shared by four trees, where diagonally opposite
  trees only touch at a corner.

The boundary faces are named `:x_neg`, `:x_pos`, `:y_neg`, and `:y_pos`.
All other keyword arguments are passed to [`p4est_mesh_from_vertices`](@ref).
"""
function p4est_mesh_five_trees(; l = 0.5, trees_per_block = 1, polydeg, kwargs...)
    @assert 0<l<1 "`l` must be in (0, 1)"
    @assert trees_per_block>=1 "`trees_per_block` must be positive"

    vertices, tree_to_vertex = five_trees_vertices(l, trees_per_block)

    return p4est_mesh_from_vertices(vertices, tree_to_vertex; polydeg,
                                    boundary_name = box_boundary_name, kwargs...)
end

function five_trees_vertices(l, trees_per_block)
    # Corners of the five blocks in p4est z-order. The trapezoids are rotated copies of the
    # one on the right, whose first reference coordinate points away from the center.
    rotate(x, n_quarter_turns) = foldl((y, _) -> SVector(-y[2], y[1]),
                                       1:n_quarter_turns; init = x)
    right = (SVector(l, -l), SVector(1.0, -1.0), SVector(l, l), SVector(1.0, 1.0))
    blocks = [
        (SVector(-l, -l), SVector(l, -l), SVector(-l, l), SVector(l, l)), # center
        right, # right
        map(x -> rotate(x, 1), right), # top
        map(x -> rotate(x, 2), right), # left
        map(x -> rotate(x, 3), right) # bottom
    ]

    n = trees_per_block
    vertex_ids = Dict{NTuple{2, Float64}, Int}()
    vertex_list = SVector{2, Float64}[]
    tree_to_vertex = Matrix{Int}(undef, 4, 5 * n^2)

    function vertex_id!(x)
        # Round to merge vertices that are computed from different blocks
        key = (round(x[1], digits = 12) + 0.0, round(x[2], digits = 12) + 0.0)
        return get!(vertex_ids, key) do
            push!(vertex_list, x)
            return length(vertex_list)
        end
    end

    tree = 0
    for block in blocks
        for b in 0:(n - 1), a in 0:(n - 1)
            tree += 1
            for (c, (da, db)) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1)))
                xi = -1 + 2 * (a + da) / n
                eta = -1 + 2 * (b + db) / n
                shape = bilinear_shape_functions(xi, eta)
                x = sum(shape[k] * block[k] for k in 1:4)
                tree_to_vertex[c, tree] = vertex_id!(x)
            end
        end
    end

    vertices = reduce(hcat, vertex_list)

    return vertices, tree_to_vertex
end

function box_boundary_name(x)
    if abs(x[1] + 1) < 1e-12
        return :x_neg
    elseif abs(x[1] - 1) < 1e-12
        return :x_pos
    elseif abs(x[2] + 1) < 1e-12
        return :y_neg
    elseif abs(x[2] - 1) < 1e-12
        return :y_pos
    else
        error("face with midpoint $x is not on the boundary of the box")
    end
end
end # @muladd
