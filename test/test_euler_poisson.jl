module TestEulerPoisson

using Test
using Trixi
using AeroTrixi
import Gridap
using OrdinaryDiffEqLowStorageRK

include("mesh_test_utils.jl")

const BOX_SIDES = (:x_neg, :x_pos, :y_neg, :y_pos)

function initial_condition_bump(x, t, equations::CompressibleEulerPoissonEquations2D)
    rho = 1 + 0.5 * exp(-20 * ((x[1] - 0.1)^2 + x[2]^2))
    p = 1 + 0.3 * exp(-30 * (x[1]^2 + (x[2] + 0.2)^2))
    # Discontinuous potential and field to check that they are not changed by the DG
    # discretization
    φ = x[1] > 0 ? 1.0 : -1.0
    return prim2cons(SVector(rho, 0.2, -0.1, p, φ, x[2], -x[1]), equations)
end

function initial_condition_bump(x, t, equations::CompressibleEulerEquations2D)
    rho = 1 + 0.5 * exp(-20 * ((x[1] - 0.1)^2 + x[2]^2))
    p = 1 + 0.3 * exp(-30 * (x[1]^2 + (x[2] + 0.2)^2))
    return prim2cons(SVector(rho, 0.2, -0.1, p), equations)
end

function shock_capturing_solver(equations)
    basis = LobattoLegendreBasis(2)
    indicator = IndicatorHennemannGassner(equations, basis, alpha_max = 0.5,
                                          alpha_min = 0.001, alpha_smooth = true,
                                          variable = density_pressure)
    return DGSEM(basis, flux_lax_friedrichs,
                 VolumeIntegralShockCapturingHG(indicator; volume_flux_dg = flux_ranocha,
                                                volume_flux_fv = flux_lax_friedrichs))
end

function rhs(equations, mesh; source_terms = nothing)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_bump,
                                        shock_capturing_solver(equations);
                                        boundary_conditions = boundary_condition_do_nothing,
                                        source_terms)
    ode = semidiscretize(semi, (0.0, 1.0))
    du_ode = similar(ode.u0)
    ode.f(du_ode, ode.u0, semi, 0.0)
    return copy(Trixi.wrap_array(du_ode, semi)), semi, ode.u0
end

@testset "Euler-Poisson" begin
    @testset "CompressibleEulerPoissonEquations2D" begin
        equations = CompressibleEulerPoissonEquations2D(1.4)
        @test nvariables(equations) == 7
        @test Trixi.varnames(cons2prim, equations) ==
              ("rho", "v1", "v2", "p", "phi", "E1", "E2")
        prim = SVector(1.2, 0.3, -0.4, 2.0, 0.5, -1.0, 2.0)
        u = prim2cons(prim, equations)
        @test cons2prim(u, equations) ≈ prim
        @test u[5:7] == prim[5:7]
        @test pressure(u, equations) ≈ 2.0
        @test density_pressure(u, equations) ≈ 2.4

        # On a mesh with hanging nodes and shock capturing, the Euler part of the right-hand
        # side is the one of the Euler equations, and φ and E are not changed
        mesh = p4est_mesh_five_trees(l = 0.4, trees_per_block = 2, polydeg = 2,
                                     initial_refinement_level = 1)
        refine_pattern!(mesh, :C, 0.4)
        du, _, _ = rhs(equations, mesh)
        du_euler, _, _ = rhs(CompressibleEulerEquations2D(1.4), mesh)
        @test maximum(abs, du[1:4, :, :, :] - du_euler) < 1e-12
        @test maximum(abs, du[5:7, :, :, :]) == 0

        # Source terms: force ρE and work ρv⋅E, scaled by the coupling
        equations_coupled = CompressibleEulerPoissonEquations2D(1.4; coupling = 2.0)
        s = source_terms_electrostatic(u, SVector(0.0, 0.0), 0.0, equations_coupled)
        rho, v1, v2 = prim[1], prim[2], prim[3]
        @test s ≈ SVector(0, 2 * rho * prim[6], 2 * rho * prim[7],
                      2 * rho * (v1 * prim[6] + v2 * prim[7]), 0, 0, 0)
        @test all(iszero,
                  source_terms_electrostatic(u, SVector(0.0, 0.0), 0.0, equations))
    end

    @testset "GridapPoissonCallback with AMR" begin
        equations = CompressibleEulerPoissonEquations2D(1.4)
        mesh = p4est_mesh_five_trees(l = 0.4, trees_per_block = 2, polydeg = 2,
                                     initial_refinement_level = 1)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_bump,
                                            shock_capturing_solver(equations);
                                            boundary_conditions = boundary_condition_do_nothing,
                                            source_terms = source_terms_electrostatic)
        ode = semidiscretize(semi, (0.0, 0.05))
        amr_indicator = IndicatorHennemannGassner(semi, alpha_max = 0.5,
                                                  alpha_min = 0.001,
                                                  alpha_smooth = true,
                                                  variable = density_pressure)
        amr_controller = ControllerThreeLevel(semi, amr_indicator, base_level = 1,
                                              med_level = 2, med_threshold = 0.001,
                                              max_level = 3, max_threshold = 0.01)
        amr_callback = AMRCallback(semi, amr_controller, interval = 2,
                                   adapt_initial_condition = true)
        source = (u, x, equations) -> SVector(u[1] - 1)
        poisson_callback = GridapPoissonCallback(semi; source,
                                                 dirichlet_tags = BOX_SIDES)
        callbacks = CallbackSet(amr_callback, poisson_callback,
                                StepsizeCallback(cfl = 0.5))
        sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
                    ode_default_options()..., callback = callbacks)
        @test sol.t[end] ≈ 0.05
        update = poisson_callback.affect!
        n_steps = sol.stats.naccept
        # One solve at the start and one after each step; rebuilds after mesh changes
        @test update.n_solves == n_steps + 1
        @test update.n_setups >= 2
        @test Trixi.ncells(mesh) > 80
        @test occursin("GridapPoissonCallback", sprint(show, poisson_callback))
        @test occursin("GridapPoissonCallback",
                       sprint(show, MIME"text/plain"(), poisson_callback))

        # φ and E of the final state equal a separate solve on the final mesh
        u_final = copy(sol.u[end])
        u_check = copy(u_final)
        Trixi.wrap_array(u_check, semi)[5:7, :, :, :] .= 0
        check = GridapPoissonCallback(semi; source, dirichlet_tags = BOX_SIDES)
        Base.get_extension(AeroTrixi, :AeroTrixiGridapExt).update_gridap_poisson!(u_check,
                                                                                  semi,
                                                                                  check.affect!)
        @test maximum(abs, u_check - u_final) < 1e-12

        # The potential is zero on the boundary and E = -∇φ is consistent with φ:
        # E points from high to low potential
        u = Trixi.wrap_array(u_final, semi)
        @test maximum(abs, u[5, :, :, :]) > 0
        @test maximum(abs, u[6:7, :, :, :]) > 0

        # Update on each Runge-Kutta stage: the callback only acts at the start and after
        # mesh changes, with a rebuild each time
        mesh = p4est_mesh_five_trees(l = 0.4, trees_per_block = 2, polydeg = 2,
                                     initial_refinement_level = 1)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_bump,
                                            shock_capturing_solver(equations);
                                            boundary_conditions = boundary_condition_do_nothing,
                                            source_terms = source_terms_electrostatic)
        ode = semidiscretize(semi, (0.0, 0.05))
        amr_indicator = IndicatorHennemannGassner(semi, alpha_max = 0.5,
                                                  alpha_min = 0.001,
                                                  alpha_smooth = true,
                                                  variable = density_pressure)
        amr_controller = ControllerThreeLevel(semi, amr_indicator, base_level = 1,
                                              med_level = 2, med_threshold = 0.001,
                                              max_level = 3, max_threshold = 0.01)
        amr_callback = AMRCallback(semi, amr_controller, interval = 2,
                                   adapt_initial_condition = true)
        poisson_callback = GridapPoissonCallback(semi; source,
                                                 dirichlet_tags = BOX_SIDES)
        stage_update = gridap_poisson_stage_update(poisson_callback)
        @test stage_update === poisson_callback.affect!
        @test occursin("stages", sprint(show, poisson_callback))
        callbacks = CallbackSet(amr_callback, poisson_callback,
                                StepsizeCallback(cfl = 0.5))
        sol = solve(ode,
                    CarpenterKennedy2N54(; stage_limiter! = stage_update,
                                         williamson_condition = false);
                    dt = 1.0, ode_default_options()..., callback = callbacks)
        @test sol.t[end] ≈ 0.05
        # Five stage updates per step (four stages and the end of the step), plus one solve
        # at the start and after each mesh change
        @test stage_update.n_setups >= 2
        @test stage_update.n_solves ==
              5 * sol.stats.naccept + stage_update.n_setups
        u_final = copy(sol.u[end])
        u_check = copy(u_final)
        Trixi.wrap_array(u_check, semi)[5:7, :, :, :] .= 0
        check = GridapPoissonCallback(semi; source, dirichlet_tags = BOX_SIDES)
        Base.get_extension(AeroTrixi, :AeroTrixiGridapExt).update_gridap_poisson!(u_check,
                                                                                  semi,
                                                                                  check.affect!)
        @test maximum(abs, u_check - u_final) < 1e-12
    end

    @testset "Order in time of the coupling" begin
        # With the force of the field on the flow, the update of φ and E once per step
        # limits the time integration to first order; the update on each stage keeps the
        # fourth order of CarpenterKennedy2N54
        function initial_condition_smooth(x, t,
                                          equations::CompressibleEulerPoissonEquations2D)
            rho = 1 + 0.5 * exp(-20 * ((x[1] - 0.1)^2 + x[2]^2))
            p = 1 + 0.3 * exp(-30 * (x[1]^2 + (x[2] + 0.2)^2))
            return prim2cons(SVector(rho, 0.2, -0.1, p, 0.0, 0.0, 0.0), equations)
        end
        function euler_variables_at_end(dt, stages)
            equations = CompressibleEulerPoissonEquations2D(1.4; coupling = 5.0)
            mesh = p4est_mesh_five_trees(l = 0.4, trees_per_block = 2, polydeg = 2,
                                         initial_refinement_level = 1)
            solver = DGSEM(polydeg = 2, surface_flux = flux_lax_friedrichs,
                           volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))
            semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_smooth,
                                                solver;
                                                boundary_conditions = boundary_condition_do_nothing,
                                                source_terms = source_terms_electrostatic)
            ode = semidiscretize(semi, (0.0, 0.1))
            poisson_callback = GridapPoissonCallback(semi;
                                                     source = (u, x, equations) -> SVector(u[1] -
                                                                                           1),
                                                     dirichlet_tags = BOX_SIDES)
            alg = if stages
                CarpenterKennedy2N54(;
                                     stage_limiter! = gridap_poisson_stage_update(poisson_callback),
                                     williamson_condition = false)
            else
                CarpenterKennedy2N54(williamson_condition = false)
            end
            sol = solve(ode, alg; dt, adaptive = false, save_everystep = false,
                        callback = CallbackSet(poisson_callback))
            return Trixi.wrap_array(sol.u[end], semi)[1:4, :, :, :]
        end
        reference = euler_variables_at_end(0.1 / 160, true)
        for (stages, order) in ((false, 1), (true, 4))
            errors = [maximum(abs, euler_variables_at_end(0.1 / n, stages) - reference)
                      for n in (10, 20)]
            @test log2(errors[1] / errors[2])≈order atol=0.2
        end
    end
end

end # module
