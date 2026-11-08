# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing-EC-Trixi for further details.
@muladd begin
#! format: noindent

"""
    p4est_mesh_forward_facing_step(; polydeg, domain_length = 3, domain_height = 1,
                                   step_position = 1, step_height = 1/3,
                                   tree_size = step_height, kwargs...)

Create a 2D `P4estMesh` of the forward-facing step channel of Woodward and Colella: the
rectangle `[0, domain_length] × [0, domain_height]` without the step
`[step_position, domain_length] × [0, step_height]` in its lower right corner. The domain is
tiled with square trees of side `tree_size`, so that all lengths must be integer multiples of
`tree_size`; the default of one third gives 21 trees for the standard geometry.

The boundary faces are named `:inflow` (left side, ``x = 0``), `:outflow` (right side,
``x = `` `domain_length`), and `:wall` (bottom, top, and the two faces of the step).
The connectivity is created with [`p4est_mesh_from_vertices`](@ref) and contains the full
face and corner connectivity. All other keyword arguments are passed to that function, in
particular `initial_refinement_level`.
"""
function p4est_mesh_forward_facing_step(; polydeg, domain_length = 3.0,
                                        domain_height = 1.0,
                                        step_position = 1.0, step_height = 1 / 3,
                                        tree_size = step_height, kwargs...)
    nx = round(Int, domain_length / tree_size)
    ny = round(Int, domain_height / tree_size)
    ix = round(Int, step_position / tree_size)
    iy = round(Int, step_height / tree_size)

    @assert nx * tree_size≈domain_length "`domain_length` must be a multiple of `tree_size`"
    @assert ny * tree_size≈domain_height "`domain_height` must be a multiple of `tree_size`"
    @assert ix * tree_size≈step_position "`step_position` must be a multiple of `tree_size`"
    @assert iy * tree_size≈step_height "`step_height` must be a multiple of `tree_size`"
    @assert 0<ix<nx "the step must lie strictly inside the domain in x"
    @assert 0<iy<ny "the step must lie strictly inside the domain in y"

    # Vertices are addressed by their integer grid position; the coordinates are computed
    # from the domain size so that the domain boundaries and the step are hit exactly.
    vertex_ids = Dict{Tuple{Int, Int}, Int}()
    vertex_list = SVector{2, Float64}[]
    function vertex_id!(i, j)
        return get!(vertex_ids, (i, j)) do
            push!(vertex_list,
                  SVector(i * domain_length / nx, j * domain_height / ny))
            return length(vertex_list)
        end
    end

    trees = NTuple{4, Int}[]
    for j in 0:(ny - 1), i in 0:(nx - 1)
        if i >= ix && j < iy
            continue # inside the step
        end
        # z-order of the corners: (-1,-1), (1,-1), (-1,1), (1,1)
        push!(trees,
              (vertex_id!(i, j), vertex_id!(i + 1, j),
               vertex_id!(i, j + 1), vertex_id!(i + 1, j + 1)))
    end

    vertices = reduce(hcat, vertex_list)
    tree_to_vertex = Matrix{Int}(undef, 4, length(trees))
    for (tree, corners) in enumerate(trees)
        tree_to_vertex[:, tree] .= corners
    end

    boundary_name = x -> forward_facing_step_boundary_name(x, domain_length)

    return p4est_mesh_from_vertices(vertices, tree_to_vertex; polydeg, boundary_name,
                                    kwargs...)
end

function forward_facing_step_boundary_name(x, domain_length)
    tol = 1e-12 * domain_length
    if abs(x[1]) < tol
        return :inflow
    elseif abs(x[1] - domain_length) < tol
        return :outflow
    else
        return :wall
    end
end
end # @muladd
