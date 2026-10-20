module TestLnodesQ2

using Test
using Trixi
using AeroTrixi
using OrdinaryDiffEqLowStorageRK

include("mesh_test_utils.jl")

# A global quadratic is represented exactly by Q2 on straight-sided bilinear elements
quadratic(x) = 1 + 2 * x[1] - 3 * x[2] + 0.5 * x[1]^2 + 1.5 * x[1] * x[2] - x[2]^2

function n_mortars(mesh)
    solver = DGSEM(polydeg = 2)
    names = Tuple(unique(filter(!=(Symbol("---")), vec(mesh.boundary_names))))
    boundary_conditions = NamedTuple{names}(map(_ -> boundary_condition_do_nothing, names))
    semi = SemidiscretizationHyperbolic(mesh, CompressibleEulerEquations2D(1.4),
                                        initial_condition_constant, solver;
                                        boundary_conditions)
    return Trixi.nmortars(semi.cache.mortars)
end

# Boundary names of a point on the boundary of the box [-1, 1]^2, sorted
function box_names(x; atol = 1e-12)
    names = Symbol[]
    isapprox(x[1], -1; atol) && push!(names, :x_neg)
    isapprox(x[1], 1; atol) && push!(names, :x_pos)
    isapprox(x[2], -1; atol) && push!(names, :y_neg)
    isapprox(x[2], 1; atol) && push!(names, :y_pos)
    return sort(names)
end

function node_names(lnodes, node)
    lnodes.node_entity[node] == 0 ? Symbol[] :
    lnodes.boundary_entities[lnodes.node_entity[node]]
end

"""
Check a `P4estLnodesQ2` against the geometry of `mesh`, and the boundary names of all
nodes against `expected_names(x)`.
"""
function check_lnodes_q2(mesh, expected_names; atol = 1e-12)
    lnodes = P4estLnodesQ2(mesh)
    @unpack element_nodes, coordinates, n_independent, n_hanging = lnodes
    n = n_independent + n_hanging
    n_elements = Trixi.ncells(mesh)
    @test size(element_nodes) == (9, n_elements)
    @test sort(unique(vec(element_nodes))) == 1:n

    # The nodes of each element coincide with the nodes of a Trixi.jl element with
    # polydeg = 2, so DG data can be transferred node by node
    nodes = LobattoLegendreBasis(2).nodes
    @test collect(nodes) ≈ [-1, 0, 1]
    element_coordinates = Array{Float64}(undef, 2, 3, 3, n_elements)
    Trixi.calc_node_coordinates!(element_coordinates, mesh, nodes)
    max_mismatch = maximum(maximum(abs,
                                   element_coordinates[:, i, j, element] -
                                   coordinates[:, element_nodes[i + 3 * (j - 1), element]])
                           for element in 1:n_elements, j in 1:3, i in 1:3)
    @test max_mismatch < atol

    # No duplicated nodes
    @test length(Set(Tuple(round.(coordinates[:, node]; digits = 10)) for node in 1:n)) ==
          n

    # Two hanging nodes per mortar (one per small element)
    @test n_hanging == 2 * n_mortars(mesh)

    # Hanging nodes lie at a quarter of the coarse face, between `a` and the midpoint `m`
    for k in 1:n_hanging
        a, m, b = (coordinates[:, lnodes.hanging_masters[i, k]] for i in 1:3)
        @test isapprox(m, (a + b) / 2; atol)
        @test isapprox(coordinates[:, n_independent + k], (3 * a + b) / 4; atol)
    end

    # The constraints reproduce a global quadratic exactly
    values = [quadratic(coordinates[:, node]) for node in 1:n]
    exact = copy(values)
    values[(n_independent + 1):end] .= NaN
    apply_hanging_constraints!(values, lnodes)
    @test isapprox(values, exact; atol = 1e-12)
    values_matrix = [exact'; 2 * exact']
    values_matrix[:, (n_independent + 1):end] .= NaN
    apply_hanging_constraints!(values_matrix, lnodes)
    @test isapprox(values_matrix, [exact'; 2 * exact']; atol = 1e-12)

    # Boundary labels; hanging nodes are always interior
    @test all(node -> node_names(lnodes, node) == expected_names(coordinates[:, node]),
              1:n)
    @test all(iszero, lnodes.node_entity[(n_independent + 1):end])
    @test allunique(lnodes.boundary_entities)

    return lnodes
end

# Standard Abaqus mesh of the box [-1, 1]^2 with 3 × 3 elements and named node sets
function write_box_inp(filename)
    xs = (-1.0, -1 / 3, 1 / 3, 1.0)
    node(i, j) = 1 + i + 4 * j
    open(filename, "w") do io
        println(io, "*Heading\n box 3 x 3\n*NODE")
        for j in 0:3, i in 0:3
            println(io, node(i, j), ", ", xs[i + 1], ", ", xs[j + 1], ", 0.0")
        end
        println(io, "******* E L E M E N T S *************")
        println(io, "*ELEMENT, type=CPS4, ELSET=Surface1")
        element = 0
        for j in 0:2, i in 0:2
            element += 1
            println(io, element, ", ", node(i, j), ", ", node(i + 1, j), ", ",
                    node(i + 1, j + 1), ", ", node(i, j + 1))
        end
        for (name, nodes) in (("Left", [node(0, j) for j in 0:3]),
                              ("Right", [node(3, j) for j in 0:3]),
                              ("Bottom", [node(i, 0) for i in 0:3]),
                              ("Top", [node(i, 3) for i in 0:3]))
            println(io, "*NSET,NSET=", name)
            println(io, join(nodes, ", "), ",")
        end
    end
    return filename
end

function inp_box_names(x; atol = 1e-12)
    names = Symbol[]
    isapprox(x[1], -1; atol) && push!(names, :Left)
    isapprox(x[1], 1; atol) && push!(names, :Right)
    isapprox(x[2], -1; atol) && push!(names, :Bottom)
    isapprox(x[2], 1; atol) && push!(names, :Top)
    return sort(names)
end

@testset "P4estLnodesQ2" begin
    @testset "five trees (l = $l, trees_per_block = $n, pattern $pattern)" for l in (0.5,
                                                                                     0.4),
                                                                               n in (1, 2),
                                                                               pattern in (:A,
                                                                                           :B,
                                                                                           :C,
                                                                                           :D,
                                                                                           :E,
                                                                                           :F)

        mesh = p4est_mesh_five_trees(; l, polydeg = 2, trees_per_block = n,
                                     initial_refinement_level = 1)
        refine_pattern!(mesh, pattern, l)
        lnodes = check_lnodes_q2(mesh, box_names)
        if pattern == :A
            @test lnodes.n_hanging == 0
        else
            @test lnodes.n_hanging > 0
        end
        # Four sides and four corners of the box
        @test sort(lnodes.boundary_entities) ==
              sort([[:x_neg], [:x_pos], [:y_neg], [:y_pos],
                       [:x_neg, :y_neg], [:x_neg, :y_pos], [:x_pos, :y_neg],
                       [:x_pos, :y_pos]])
        # A side with k element faces has 2k - 1 nodes that only carry its name
        n_side_nodes = count(node -> node_names(lnodes, node) == [:x_pos],
                             1:(lnodes.n_independent + lnodes.n_hanging))
        @test count(face -> face[3] == :x_pos, lnodes.boundary_faces) ==
              (n_side_nodes + 1) ÷ 2
    end

    @testset "Abaqus mesh with named node sets" begin
        mktempdir() do dir
            meshfile = write_box_inp(joinpath(dir, "box.inp"))
            mesh = P4estMesh{2}(meshfile; polydeg = 2, initial_refinement_level = 1,
                                boundary_symbols = [:Left, :Right, :Bottom, :Top])
            refine_where!(mesh, x -> x[1] > 0 && x[2] > 0; levels = 2)
            lnodes = check_lnodes_q2(mesh, inp_box_names)
            @test lnodes.n_hanging > 0
            @test length(lnodes.boundary_entities) == 8
            @test [:Bottom, :Left] in lnodes.boundary_entities
        end
    end

    @testset "after a simulation with AMR" begin
        equations = CompressibleEulerEquations2D(1.4)
        function initial_condition_blast(x, t, equations)
            r = sqrt(x[1]^2 + x[2]^2)
            rho, p = r > 0.5 ? (1.0, 1.0e-1) : (1.1691, 1.245)
            return prim2cons(SVector(rho, 0.0, 0.0, p), equations)
        end
        basis = LobattoLegendreBasis(2)
        indicator_sc = IndicatorHennemannGassner(equations, basis, alpha_max = 0.5,
                                                 alpha_min = 0.001, alpha_smooth = true,
                                                 variable = density_pressure)
        volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                         volume_flux_dg = flux_ranocha,
                                                         volume_flux_fv = flux_lax_friedrichs)
        solver = DGSEM(basis, flux_lax_friedrichs, volume_integral)
        mesh = p4est_mesh_five_trees(l = 0.4, trees_per_block = 2, polydeg = 2,
                                     initial_refinement_level = 1)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_blast,
                                            solver;
                                            boundary_conditions = boundary_condition_do_nothing)
        ode = semidiscretize(semi, (0.0, 0.05))
        amr_indicator = IndicatorHennemannGassner(semi, alpha_max = 0.5, alpha_min = 0.001,
                                                  alpha_smooth = true,
                                                  variable = density_pressure)
        amr_controller = ControllerThreeLevel(semi, amr_indicator, base_level = 1,
                                              med_level = 2, med_threshold = 0.05,
                                              max_level = 3, max_threshold = 0.2)
        amr_callback = AMRCallback(semi, amr_controller, interval = 2,
                                   adapt_initial_condition = true)
        callbacks = CallbackSet(amr_callback, StepsizeCallback(cfl = 0.5))
        sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
                    ode_default_options()..., callback = callbacks)
        @test sol.t[end] ≈ 0.05
        lnodes = check_lnodes_q2(mesh, box_names)
        @test lnodes.n_hanging > 0
        @test Trixi.ncells(mesh) > 80
    end
end

end # module
