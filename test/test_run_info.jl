module TestRunInfo

using Test
using Trixi
using AeroTrixi: RunInfo, @RunInfo, find_runs
using OrdinaryDiffEqLowStorageRK
using TOML

# a small, fast problem to run
function small_semi()
    equations = LinearScalarAdvectionEquation2D((1.0, 1.0))
    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level = 2,
                    periodicity = true)
    return SemidiscretizationHyperbolic(mesh, equations,
                                        initial_condition_convergence_test,
                                        DGSEM(polydeg = 3),
                                        boundary_conditions = boundary_condition_periodic)
end

function run_small(semi, run_info)
    ode = semidiscretize(semi, (0.0, 0.1))
    return solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                 dt = 0.01, save_everystep = false, callback = run_info.callback)
end

@testset "RunInfo" begin
    root = mktempdir()
    semi = small_semi()

    @testset "directory and description" begin
        run_info = @RunInfo(semi; parameters = (; polydeg = 3, cfl = 0.5),
                            output_root = root)
        @test run_info.elixir == "test_run_info"
        @test run_info.label == "polydeg=3__cfl=0.5"
        @test run_info.dir == joinpath(root, "test_run_info", "polydeg=3__cfl=0.5")
        @test isfile(joinpath(run_info.dir, "test_run_info.jl"))
        @test isfile(joinpath(run_info.dir, "setup.txt"))
        toml = TOML.parsefile(joinpath(run_info.dir, "run.toml"))
        @test toml["run"]["status"] == "created"
        @test toml["parameters"] == Dict("polydeg" => 3, "cfl" => 0.5)

        sol = run_small(semi, run_info)
        toml = TOML.parsefile(joinpath(run_info.dir, "run.toml"))
        @test toml["run"]["status"] == "finished"
        @test toml["result"]["final_time"] ≈ 0.1
        @test toml["result"]["timesteps"] == 10
        @test toml["result"]["n_elements"] == 16
        @test isfile(joinpath(run_info.dir, toml["files"]["restart"]))
        @test isfile(joinpath(run_info.dir, toml["files"]["mesh"]))
        @test Trixi.load_time(joinpath(run_info.dir, toml["files"]["restart"])) ≈ 0.1
    end

    @testset "find_runs" begin
        other = @RunInfo(semi; parameters = (; polydeg = 4, cfl = 0.5),
                         output_root = root)
        # created, but not run: only found when asking for all states
        @test find_runs(root) == [joinpath(root, "test_run_info", "polydeg=3__cfl=0.5")]
        @test length(find_runs(root; status = nothing)) == 2
        run_small(semi, other)
        @test length(find_runs(root)) == 2
        @test find_runs(root; polydeg = 4) == [other.dir]
        @test isempty(find_runs(root; polydeg = 5))
        @test isempty(find_runs(joinpath(root, "does_not_exist")))
    end

    @testset "the same parameters replace a previous run" begin
        dir = joinpath(root, "test_run_info", "polydeg=3__cfl=0.5")
        touch(joinpath(dir, "old_result.txt"))
        run_info = @RunInfo(semi; parameters = (; polydeg = 3, cfl = 0.5),
                            output_root = root)
        @test run_info.dir == dir
        @test !isfile(joinpath(dir, "old_result.txt"))
    end

    @testset "directories not created by RunInfo are not deleted" begin
        dir = joinpath(root, "test_run_info", "mine")
        mkpath(dir)
        touch(joinpath(dir, "important.txt"))
        @test_throws ErrorException @RunInfo(semi; name = "mine", output_root = root)
        @test isfile(joinpath(dir, "important.txt"))
    end

    @testset "explicit name and missing source" begin
        run_info = RunInfo(semi; source = "does_not_exist.jl", name = "my run",
                           output_root = root)
        # the file being included is used instead of the missing source
        @test run_info.elixir == "test_run_info"
        @test run_info.label == "my-run"
    end
end

end # module
