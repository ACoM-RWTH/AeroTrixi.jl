# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing-EC-Trixi for further details.
@muladd begin
#! format: noindent

"""
    RelaxationStepsizeCallback(; nu_dt_max = 1.0)

Limit the time step of an explicit time integration by the stiffness of the collision terms
of the [`Grad13Equations2D`](@ref): after the `StepsizeCallback` has set the time step from
the CFL condition, this callback reduces it such that ``\\nu_\\mathrm{max} \\Delta t \\le``
`nu_dt_max`, with the maximum of the [`collision_frequency`](@ref) over all nodes of the
solution. The callback must come after the `StepsizeCallback` in the `CallbackSet`.

The value of `nu_dt_max` should stay below the real-axis stability limit of the Runge-Kutta
method (about 2.5 for the third-order SSP method and the five-stage fourth-order
Carpenter-Kennedy method, and 2 for Heun's method); the default of 1 also keeps the
relaxation resolved in time.
"""
struct RelaxationStepsizeCallback{RealT <: Real}
    nu_dt_max::RealT
end

function RelaxationStepsizeCallback(; nu_dt_max = 1.0)
    relaxation_stepsize_callback = RelaxationStepsizeCallback{typeof(nu_dt_max)}(nu_dt_max)

    return DiscreteCallback(relaxation_stepsize_callback, relaxation_stepsize_callback, # the first one is the condition, the second the affect!
                            save_positions = (false, false),
                            initialize = initialize_relaxation_stepsize!)
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:RelaxationStepsizeCallback})
    @nospecialize cb # reduce precompilation time

    relaxation_stepsize_callback = cb.affect!
    print(io, "RelaxationStepsizeCallback(nu_dt_max=",
          relaxation_stepsize_callback.nu_dt_max,
          ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:RelaxationStepsizeCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        relaxation_stepsize_callback = cb.affect!

        setup = [
            "max ν⋅Δt" => relaxation_stepsize_callback.nu_dt_max
        ]
        summary_box(io, "RelaxationStepsizeCallback", setup)
    end
    return nothing
end

function initialize_relaxation_stepsize!(cb, u, t, integrator)
    cb.affect!(integrator)
    return nothing
end

# this method is called to determine whether the callback should be activated
function (relaxation_stepsize_callback::RelaxationStepsizeCallback)(u, t, integrator)
    return true
end

# This method is called as callback during the time integration.
@inline function (relaxation_stepsize_callback::RelaxationStepsizeCallback)(integrator)
    if integrator.opts.adaptive
        throw(ArgumentError("The `RelaxationStepsizeCallback` has no effect when using an adaptive time integration scheme. Please remove the `RelaxationStepsizeCallback` or set `adaptive = false` in `solve`."))
    end

    semi = integrator.p
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(integrator.u, mesh, equations, solver, cache)

    nu_max = @trixi_timeit timer() "calculate dt" max_collision_frequency(u, mesh,
                                                                          equations,
                                                                          solver, cache)
    dt_relaxation = relaxation_stepsize_callback.nu_dt_max / nu_max
    dt = min(get_proposed_dt(integrator), dt_relaxation)

    set_proposed_dt!(integrator, dt)
    integrator.opts.dtmax = dt
    integrator.dtcache = dt

    # avoid re-evaluating possible FSAL stages
    derivative_discontinuity!(integrator, false)
    return nothing
end
end # @muladd
