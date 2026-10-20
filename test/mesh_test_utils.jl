# Helpers to refine `P4estMesh`es in tests, shared by several test files

using Random: MersenneTwister
using Trixi

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
