# The quantities computed along surfaces and their specialized implementations
# for some solvers
include("analysis_surface_integral.jl")
include("analysis_surface_pointwise.jl")

"""
    SurfacePointwiseCallback(semi, quantities::AnalysisSurfacePointwise...; interval)

Compute and save pointwise surface quantities, such as the
[`SurfacePressureCoefficient`](@ref) or the [`SurfaceFrictionCoefficient`](@ref),
at the start of the simulation, every `interval` time steps, and at the end.
Each quantity is given as an [`AnalysisSurfacePointwise`](@ref), which determines the
boundaries and the output directory; the values at all surface nodes are written to
`<output directory>/<variable name>_<time step>.h5`, e.g., `out/CP_x_000100.h5`.

The quantities are computed from an evaluation of the right-hand side at the current
time, which provides the solution at the boundaries and, for viscous quantities such
as the friction coefficient, the gradients of the solution.

This callback replaces the keyword argument `analysis_pointwise` of the
`AnalysisCallback` of AeroTrixi.jl v0.4 and is used together with the
`AnalysisCallback` of Trixi.jl.

# Example
```julia
pressure_coefficient = AnalysisSurfacePointwise((:airfoil,),
                                                SurfacePressureCoefficient(p_inf, rho_inf,
                                                                           u_inf))
surface_pointwise_callback = SurfacePointwiseCallback(semi, pressure_coefficient;
                                                      interval = 100)
callbacks = CallbackSet(summary_callback, analysis_callback, surface_pointwise_callback)
```
"""
struct SurfacePointwiseCallback{Quantities}
    interval::Int
    quantities::Quantities

    # only the parametric constructor, so that there is no ambiguity with the
    # user-facing constructor below
    function SurfacePointwiseCallback{Quantities}(interval,
                                                  quantities) where {Quantities}
        return new{Quantities}(interval, quantities)
    end
end

function SurfacePointwiseCallback(semi, quantities::AnalysisSurfacePointwise...;
                                  interval)
    surface_pointwise = SurfacePointwiseCallback{typeof(quantities)}(interval, quantities)

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    condition = (u, t, integrator) -> interval > 0 &&
        (integrator.stats.naccept % interval == 0 || isfinished(integrator))

    return DiscreteCallback(condition, surface_pointwise,
                            save_positions = (false, false),
                            initialize = initialize_surface_pointwise!)
end

# Also save the quantities at the start of the simulation
function initialize_surface_pointwise!(cb, u, t, integrator)
    cb.affect!(integrator)
    return nothing
end

function (surface_pointwise::SurfacePointwiseCallback)(integrator)
    semi = integrator.p
    u_ode = integrator.u
    du_ode = first(get_tmp_cache(integrator))

    @trixi_timeit timer() "surface pointwise" begin
        # The surface quantities use the solution at the boundaries and, for viscous
        # quantities, the gradients, which are computed when evaluating the
        # right-hand side. `integrator.f` is usually just a call to `rhs!`.
        @notimeit timer() integrator.f(du_ode, u_ode, semi, integrator.t)

        mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
        u = wrap_array(u_ode, mesh, equations, solver, cache)
        du = wrap_array(du_ode, mesh, equations, solver, cache)
        analyze_pointwise(surface_pointwise.quantities, du, u, integrator.t, semi,
                          integrator.stats.naccept)
    end

    # avoid re-evaluating possible FSAL stages
    derivative_discontinuity!(integrator, false)
    return nothing
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SurfacePointwiseCallback})
    @nospecialize cb # reduce precompilation time
    surface_pointwise = cb.affect!
    print(io, "SurfacePointwiseCallback(interval=", surface_pointwise.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SurfacePointwiseCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        surface_pointwise = cb.affect!
        setup = Pair{String, Any}["interval" => surface_pointwise.interval]
        for (idx, quantity) in enumerate(surface_pointwise.quantities)
            push!(setup, "│ quantity " * string(idx) => quantity)
            push!(setup, "│ │ boundaries" => quantity.boundary_symbols)
            push!(setup,
                  "│ │ output directory" => abspath(normpath(quantity.output_directory)))
        end
        summary_box(io, "SurfacePointwiseCallback", setup)
    end
end

# Iterate over tuples of pointwise analysis quantities in a type-stable way using "lispy tuple programming".
function analyze_pointwise(analysis_quantities::NTuple{N, Any}, du, u, t,
                           semi, iter) where {N}

    # Extract the first pointwise analysis quantity and process it; keep the remaining to be processed later
    quantity = first(analysis_quantities)
    remaining_quantities = Base.tail(analysis_quantities)

    analyze(quantity, du, u, t, semi, iter)

    # Recursively call this method with the unprocessed pointwise analysis quantities
    analyze_pointwise(remaining_quantities, du, u, t, semi, iter)
    return nothing
end

# terminate the type-stable iteration over tuples
function analyze_pointwise(analysis_quantities::Tuple{}, du, u, t, semi, iter)
    nothing
end

# This version of `analyze` is used for `AnalysisSurfacePointwise` such as `SurfacePressureCoefficient`.
# We need the iteration number `iter` to be passed in here
# as for `AnalysisSurfacePointwise` the writing to disk is handled by the callback itself.
function analyze(quantity::AnalysisSurfacePointwise{Variable},
                 du, u, t,
                 semi::AbstractSemidiscretization,
                 iter) where {Variable}
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    # Call the `Variable`-specific `analyze` function
    analyze(quantity, du, u, t, mesh, equations, solver, cache, semi, iter)
end

# Special analyze for `SemidiscretizationHyperbolicParabolic` such that
# precomputed gradients are available. Required for `AnalysisSurfacePointwise` equipped
# with `VariableViscous` such as `SurfaceFrictionCoefficient`.
# As for the inviscid version, we need to pass in the iteration number `iter` as
# for `AnalysisSurfacePointwise` the writing to disk is handled by the callback itself.
function analyze(quantity::AnalysisSurfacePointwise{Variable},
                 du, u, t,
                 semi::SemidiscretizationHyperbolicParabolic,
                 iter) where {Variable <: VariableViscous}
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    equations_parabolic = semi.equations_parabolic
    cache_parabolic = semi.cache_parabolic
    # Call the `Variable`-specific `analyze` function
    analyze(quantity, du, u, t, mesh, equations, equations_parabolic, solver, cache, semi,
            cache_parabolic, iter)
end

"""
    AeroTrixi.AnalysisCallback(args...; kwargs...)

Removed in AeroTrixi.jl v0.5. Use the `AnalysisCallback` of Trixi.jl for errors and
integrals, and a [`SurfacePointwiseCallback`](@ref) for the pointwise surface quantities
that were passed as `analysis_pointwise` before.
"""
function AnalysisCallback(args...; kwargs...)
    throw(ArgumentError("`AeroTrixi.AnalysisCallback` was removed in AeroTrixi.jl v0.5. " *
                        "Use the `AnalysisCallback` of Trixi.jl for errors and " *
                        "integrals, and `SurfacePointwiseCallback(semi, quantities...; " *
                        "interval)` for the pointwise surface quantities passed as " *
                        "`analysis_pointwise` before."))
end
