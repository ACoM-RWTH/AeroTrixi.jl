using OrdinaryDiffEqSSPRK
using Trixi
using AeroTrixi

###############################################################################
# Modified Sod shock tube from Trixi.jl (`elixir_euler_modified_sod.jl`), using the
# AUSM+ flux of Liou (1996) as surface flux.

RealT = Float64
equations = CompressibleEulerEquations1D(RealT(1.4))

"""
    initial_condition_modified_sod(x, t, equations::CompressibleEulerEquations1D)

Modiﬁed Sod shock tube problem, presented in Section 6.4 of Toro's book.
This problem consists of a left sonic rarefaction wave and is useful for testing whether numerical solutions
violate the entropy condition.
An entropy-satisfying solution should produce a smooth(!) rarefaction wave.

## References
- Toro (2009).
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, 3rd Edition.
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

- Lin, Chan (2024)
  High order entropy stable discontinuous Galerkin spectral element methods through subcell limiting
  [arXiv: 2306.12663](https://arxiv.org/abs/2306.12663)
  [DOI: 10.1016/j.jcp.2023.112677](https://doi.org/10.1016/j.jcp.2023.112677)
"""
function initial_condition_modified_sod(x, t, equations::CompressibleEulerEquations1D)
    RealT = eltype(x)
    if x[1] < RealT(0.3)
        return prim2cons(SVector(one(RealT), 0.75f0, one(RealT)), equations)
    else
        return prim2cons(SVector(0.125f0, zero(RealT), RealT(0.1)), equations)
    end
end
initial_condition = initial_condition_modified_sod

# Shock capturing: blend the entropy-conservative flux-differencing volume integral (which
# recovers the rarefaction wave) with a subcell finite volume scheme using AUSM+
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = Trixi.density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = flux_ranocha,
                                                 volume_flux_fv = flux_ausmplus)

solver = DGSEM(basis, flux_ausmplus, volume_integral)

coordinates_min = zero(RealT)
coordinates_max = one(RealT)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6, RealT = RealT,
                periodicity = false)

# Dirichlet boundary condition is only valid for considered time interval.
# If the rarefaction wave reaches the boundary, this condition is no longer valid!
boundary_conditions = (; x_neg = BoundaryConditionDirichlet(initial_condition),
                       x_pos = boundary_condition_do_nothing)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (zero(RealT), RealT(0.2))
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)

analysis_interval = 1000
# qualified because both Trixi and AeroTrixi export an `AnalysisCallback`; AeroTrixi's
# adds surface pointwise analysis, which is not used here
analysis_callback = Trixi.AnalysisCallback(semi, interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

ode_alg = SSPRK43(thread = Trixi.Threaded())

sol = solve(ode, ode_alg;
            dt = 4e-4, adaptive = true,
            ode_default_options()..., callback = callbacks);
