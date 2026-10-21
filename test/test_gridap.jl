module TestGridap

using Test
using Trixi
using AeroTrixi
using Gridap

include("mesh_test_utils.jl")

const BOX_SIDES = (:x_neg, :x_pos, :y_neg, :y_pos)

# A global quadratic is in the Q2 space on straight-sided bilinear elements, so the
# Galerkin solution of the Poisson problem is exact
u_quadratic(x) = 1 + 2 * x[1] - 3 * x[2] + 0.5 * x[1]^2 + 1.5 * x[1] * x[2] - x[2]^2
f_quadratic(x) = 1.0 # -Δu

function errors(u, uh, dΩ)
    (sqrt(sum(∫((u - uh) * (u - uh)) * dΩ)),
     sqrt(sum(∫(∇(u - uh) ⋅ ∇(u - uh)) * dΩ)))
end

function solve_poisson(lnodes, u, f; dirichlet_tags = BOX_SIDES,
                       neumann_tags = setdiff(BOX_SIDES, dirichlet_tags),
                       constrained = true)
    model = gridap_model(lnodes)
    q = gridap_q2_space(lnodes, model; dirichlet_tags)
    V = constrained ? q.space : q.unconstrained
    U = TrialFESpace(V, u)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 6)
    a(u, v) = ∫(∇(v) ⋅ ∇(u)) * dΩ
    if isempty(neumann_tags)
        l = v -> ∫(v * f) * dΩ
    else
        Γ = BoundaryTriangulation(model, tags = String.(collect(neumann_tags)))
        dΓ = Measure(Γ, 6)
        n = get_normal_vector(Γ)
        l = v -> ∫(v * f) * dΩ + ∫(v * (n ⋅ ∇(u))) * dΓ
    end
    uh = solve(AffineFEOperator(a, l, U, V))
    return uh, dΩ, model, q
end

function five_tree_mesh(pattern; l = 0.4, trees_per_block = 2,
                        initial_refinement_level = 1)
    mesh = p4est_mesh_five_trees(; l, polydeg = 2, trees_per_block,
                                 initial_refinement_level)
    refine_pattern!(mesh, pattern, l)
    return mesh
end

@testset "Gridap.jl coupling" begin
    @testset "model and labels (pattern $pattern)" for pattern in (:A, :C, :F)
        lnodes = P4estLnodesQ2(five_tree_mesh(pattern))
        model = gridap_model(lnodes)
        @test num_cells(model) == size(lnodes.element_nodes, 2)
        labels = get_face_labeling(model)
        for tag in ("interior", "boundary", String.(BOX_SIDES)...)
            @test tag in labels.tag_to_name
        end
        # Faces at hanging nodes are interior faces
        length_of(tags) = sum(∫(1) * Measure(BoundaryTriangulation(model; tags), 2))
        @test length_of("boundary") ≈ 8
        for side in BOX_SIDES
            @test length_of(String(side)) ≈ 2
        end
        @test sum(∫(1) * Measure(Triangulation(model), 2)) ≈ 4

        # Free and Dirichlet degrees of freedom
        q = gridap_q2_space(lnodes, model; dirichlet_tags = BOX_SIDES)
        n_nodes = lnodes.n_independent + lnodes.n_hanging
        n_boundary = count(!iszero, lnodes.node_entity)
        @test num_dirichlet_dofs(q.unconstrained) == n_boundary
        @test num_free_dofs(q.unconstrained) == n_nodes - n_boundary
        @test num_free_dofs(q.space) == n_nodes - n_boundary - lnodes.n_hanging
        @test count(<(0), q.node_to_dof) == n_boundary
    end

    @testset "exact quadratic, Dirichlet (pattern $pattern)" for pattern in (:A, :B, :C,
                                                                             :D, :E,
                                                                             :F)
        lnodes = P4estLnodesQ2(five_tree_mesh(pattern))
        uh, dΩ = solve_poisson(lnodes, u_quadratic, f_quadratic)
        e_l2, e_h1 = errors(u_quadratic, uh, dΩ)
        @test e_l2 < 1e-12
        @test e_h1 < 1e-11
    end

    @testset "hanging nodes need constraints" begin
        lnodes = P4estLnodesQ2(five_tree_mesh(:C))
        @test lnodes.n_hanging > 0
        uh, dΩ = solve_poisson(lnodes, u_quadratic, f_quadratic; constrained = false)
        @test errors(u_quadratic, uh, dΩ)[1] > 1e-6
    end

    @testset "exact quadratic, Dirichlet on x_neg and Neumann elsewhere" begin
        lnodes = P4estLnodesQ2(five_tree_mesh(:F))
        uh, dΩ = solve_poisson(lnodes, u_quadratic, f_quadratic;
                               dirichlet_tags = (:x_neg,))
        e_l2, e_h1 = errors(u_quadratic, uh, dΩ)
        @test e_l2 < 1e-11
        @test e_h1 < 1e-10
    end

    @testset "convergence with hanging nodes" begin
        u(x) = sin(π * x[1]) * sin(π * x[2])
        f(x) = 2 * π^2 * u(x)
        e = map((1, 2)) do level
            lnodes = P4estLnodesQ2(five_tree_mesh(:C; initial_refinement_level = level))
            uh, dΩ = solve_poisson(lnodes, u, f)
            errors(u, uh, dΩ)
        end
        # Q2: third order in L2, second order in H1
        @test log2(e[1][1] / e[2][1]) > 2.7
        @test log2(e[1][2] / e[2][2]) > 1.8
    end

    @testset "Abaqus mesh with named node sets" begin
        mktempdir() do dir
            meshfile = joinpath(dir, "box.inp")
            xs = (-1.0, -1 / 3, 1 / 3, 1.0)
            node(i, j) = 1 + i + 4 * j
            open(meshfile, "w") do io
                println(io, "*Heading\n box 3 x 3\n*NODE")
                for j in 0:3, i in 0:3
                    println(io, node(i, j), ", ", xs[i + 1], ", ", xs[j + 1], ", 0.0")
                end
                println(io, "******* E L E M E N T S *************")
                println(io, "*ELEMENT, type=CPS4, ELSET=Surface1")
                for (element, (i, j)) in enumerate((i, j) for j in 0:2 for i in 0:2)
                    println(io, element, ", ", node(i, j), ", ", node(i + 1, j), ", ",
                            node(i + 1, j + 1), ", ", node(i, j + 1))
                end
                for (name, nodes) in (("Left", [node(0, j) for j in 0:3]),
                                      ("Right", [node(3, j) for j in 0:3]),
                                      ("Bottom", [node(i, 0) for i in 0:3]),
                                      ("Top", [node(i, 3) for i in 0:3]))
                    println(io, "*NSET,NSET=", name, "\n", join(nodes, ", "), ",")
                end
            end
            mesh = P4estMesh{2}(meshfile; polydeg = 2, initial_refinement_level = 1,
                                boundary_symbols = [:Left, :Right, :Bottom, :Top])
            refine_where!(mesh, x -> x[1] > 0 && x[2] > 0; levels = 2)
            lnodes = P4estLnodesQ2(mesh)
            @test lnodes.n_hanging > 0
            uh, dΩ = solve_poisson(lnodes, u_quadratic, f_quadratic;
                                   dirichlet_tags = (:Left, :Bottom),
                                   neumann_tags = (:Right, :Top))
            e_l2, e_h1 = errors(u_quadratic, uh, dΩ)
            @test e_l2 < 1e-11
            @test e_h1 < 1e-10
        end
    end
end

end # module
