module TestSurfacePointwise

using Test
using Trixi
using Trixi: h5open, attributes
using AeroTrixi
using OrdinaryDiffEqLowStorageRK

# Uniform flow along the x-direction in the unit square, with slip walls at the bottom and
# the top. The flow stays uniform, so the pressure coefficient vanishes on the walls.
const RHO_INF, V_INF, P_INF = 1.0, 0.5, 1.0

function initial_condition_uniform(x, t, equations)
    prim2cons(SVector(RHO_INF, V_INF, 0.0,
                      P_INF), equations)
end

function uniform_flow_semi(; viscous = false)
    equations = CompressibleEulerEquations2D(1.4)
    mesh = P4estMesh((2, 2), polydeg = 3, coordinates_min = (0.0, 0.0),
                     coordinates_max = (1.0, 1.0), initial_refinement_level = 0,
                     periodicity = false)
    solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
    freestream = BoundaryConditionDirichlet(initial_condition_uniform)
    boundary_conditions = (; x_neg = freestream, x_pos = freestream,
                           y_neg = boundary_condition_slip_wall,
                           y_pos = boundary_condition_slip_wall)
    if !viscous
        return SemidiscretizationHyperbolic(mesh, equations, initial_condition_uniform,
                                            solver; boundary_conditions)
    end
    equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = 1.0e-2,
                                                              Prandtl = 0.72)
    wall = BoundaryConditionNavierStokesWall(NoSlip((x, t, equations) -> SVector(0.0,
                                                                                 0.0)),
                                             Adiabatic((x, t, equations) -> 0.0))
    freestream_parabolic = BoundaryConditionDirichlet((x, t, _) -> initial_condition_uniform(x,
                                                                                             t,
                                                                                             equations))
    boundary_conditions_parabolic = (; x_neg = freestream_parabolic,
                                     x_pos = freestream_parabolic,
                                     y_neg = wall, y_pos = wall)
    return SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition_uniform, solver;
                                                 boundary_conditions = (boundary_conditions,
                                                                        boundary_conditions_parabolic))
end

function run_with(semi, callback)
    ode = semidiscretize(semi, (0.0, 0.1))
    return solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 0.01,
                 save_everystep = false, callback = callback)
end

function read_output(file)
    h5open(file) do f
        (; n_points = read(attributes(f)["n_points"]), data = read(f["point_data"]),
         element_indices = read(f["element_indices"]),
         node_counter = read(f["node_counter"]),
         coordinates = read(f["point_coordinates"]), timestep = read(f["timestep"]))
    end
end

@testset "SurfacePointwiseCallback" begin
    @testset "pressure coefficient" begin
        dir = mktempdir()
        semi = uniform_flow_semi()
        cp = AnalysisSurfacePointwise((:y_neg,),
                                      SurfacePressureCoefficient(P_INF, RHO_INF, V_INF),
                                      dir)
        callback = SurfacePointwiseCallback(semi, cp; interval = 4)
        @test occursin("SurfacePointwiseCallback", sprint(show, callback))
        @test occursin("SurfacePointwiseCallback",
                       sprint(show, MIME"text/plain"(), callback))

        sol = run_with(semi, callback)
        # at the start, every 4 steps, and at the end; the number of steps may differ
        # by one due to round-off in the time steps
        final = sol.stats.naccept
        @test final in (10, 11)
        final_file = "CP_x_" * lpad(final, 6, '0') * ".h5"
        @test sort(readdir(dir)) ==
              ["CP_x_000000.h5", "CP_x_000004.h5", "CP_x_000008.h5", final_file]
        output = read_output(joinpath(dir, final_file))
        # 2 boundary elements with 4 nodes each
        @test output.n_points == 8
        @test length(output.data) == length(output.element_indices) ==
              length(output.node_counter) == output.n_points
        @test output.node_counter == 1:8
        @test all(y -> isapprox(y, 0.0, atol = 1.0e-14), output.coordinates[:, 2])
        @test all(c -> isapprox(c, 0.0, atol = 1.0e-10), output.data)
        @test output.timestep == final
    end

    @testset "friction and pressure coefficient of viscous flow" begin
        dir = mktempdir()
        semi = uniform_flow_semi(viscous = true)
        cf = AnalysisSurfacePointwise((:y_neg, :y_pos),
                                      SurfaceFrictionCoefficient(RHO_INF, V_INF), dir)
        cp = AnalysisSurfacePointwise((:y_neg,),
                                      SurfacePressureCoefficient(P_INF, RHO_INF, V_INF),
                                      dir)
        sol = run_with(semi, SurfacePointwiseCallback(semi, cf, cp; interval = 100))
        final = lpad(sol.stats.naccept, 6, '0')
        @test sort(readdir(dir)) == ["CF_x_000000.h5", "CF_x_$final.h5",
            "CP_x_000000.h5", "CP_x_$final.h5"]
        output = read_output(joinpath(dir, "CF_x_$final.h5"))
        @test output.n_points == 16
        @test length(output.element_indices) == output.n_points
        @test all(isfinite, output.data)
        # the no-slip walls slow down the flow, which causes friction
        @test maximum(abs, output.data) > 0
    end

    @testset "removed AnalysisCallback of AeroTrixi.jl" begin
        @test_throws ArgumentError AeroTrixi.AnalysisCallback(uniform_flow_semi();
                                                              analysis_pointwise = ())
    end
end

end # module
