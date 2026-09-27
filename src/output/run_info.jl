"""
    RunInfo

Record of a single simulation run, created with [`@RunInfo`](@ref).

All output of the run goes to the directory `run_info.dir`, which is
`<output_root>/<name of the elixir>/<run label>`. The run label is built from the
parameters passed to `@RunInfo`, so runs with different parameters are stored in
different directories, while running the same parameters again replaces the
previous results.

The directory contains
- `run.toml`: a description of the run (elixir, parameters, versions, status and results),
- `setup.txt`: the summary of the semidiscretization printed by Trixi.jl,
- a copy of the elixir and of the additional `files` passed to `@RunInfo`,
- the final solution as a Trixi.jl restart file and the corresponding mesh file,
  unless `save_final_state = false`.

`run_info.callback` has to be added to the `CallbackSet` of the simulation; it
updates `run.toml` when the simulation starts and ends and saves the final solution.
Use [`find_runs`](@ref) and [`load_run`](@ref) to work with the stored runs later.
"""
mutable struct RunInfo
    elixir::String
    source::Union{String, Nothing}
    parameters::Vector{Pair{String, Any}}
    label::String
    dir::String
    output_root::String
    files::Vector{String}
    save_final_state::Bool
    toml::Dict{String, Any}
    start_time::Float64
    callback::Any
end

"""
    @RunInfo(semi; parameters = (;), files = String[], name = nothing,
             output_root = "out", save_final_state = true)

Create a [`RunInfo`](@ref) for the simulation of the semidiscretization `semi`
that is set up in the current file.

The macro records the file it is called from, which works in all ways of running an
elixir: `julia elixir.jl`, `include`, `trixi_include`, and the inline evaluation in
VS Code. Without a file, e.g., when typed in the REPL, the run is stored as
`"interactive"`.

- `parameters`: a `NamedTuple` of the parameters that distinguish runs of this
  elixir, e.g., `(; polydeg, mesh_size)`. They are stored in `run.toml` and form
  the name of the output directory.
- `files`: further files to copy into the output directory, e.g., a mesh geometry file.
- `name`: use this name for the output directory instead of the parameters.
- `output_root`: the directory in which the output of all elixirs is collected.
- `save_final_state`: save the final solution as a restart file.

# Example
```julia
run_info = @RunInfo(semi; parameters = (; polydeg, mesh_size))
callbacks = CallbackSet(summary_callback, stepsize_callback, run_info.callback)
```
"""
macro RunInfo(args...)
    file = __source__.file === nothing ? "" : String(__source__.file)
    positional = Any[]
    keywords = Any[Expr(:kw, :source, file)]
    for arg in args
        if arg isa Expr && arg.head === :parameters
            append!(keywords, arg.args)
        elseif arg isa Expr && arg.head === :(=)
            push!(keywords, Expr(:kw, arg.args...))
        else
            push!(positional, arg)
        end
    end
    return esc(Expr(:call, GlobalRef(@__MODULE__, :RunInfo),
                    Expr(:parameters, keywords...), positional...))
end

function RunInfo(semi; source = nothing, parameters = (;), files = String[],
                 name = nothing, output_root = "out", save_final_state = true)
    source_file = detect_source_file(source)
    elixir = source_file === nothing ? "interactive" : splitext(basename(source_file))[1]
    params = Pair{String, Any}[string(key) => toml_value(value)
                               for (key, value) in pairs(parameters)]
    label = run_label(params, name, source_file)
    dir = joinpath(output_root, elixir, label)

    run_info = RunInfo(elixir, source_file, params, label, dir, output_root,
                       String[string(file) for file in files], save_final_state,
                       Dict{String, Any}(), NaN, nothing)
    run_info.callback = DiscreteCallback(run_info, run_info,
                                         save_positions = (false, false),
                                         initialize = initialize_run_info!,
                                         finalize = finalize_run_info!)
    prepare_run_directory!(run_info, semi)
    return run_info
end

# Find the elixir: the file recorded by the macro, the file currently being included,
# or the script passed to `julia`. Returns `nothing` if there is no such file.
function detect_source_file(source)
    if source isa AbstractString && isfile(source)
        return abspath(source)
    end
    path = Base.source_path(nothing)
    if path !== nothing && isfile(path)
        return abspath(path)
    end
    if !isempty(PROGRAM_FILE) && isfile(PROGRAM_FILE)
        return abspath(PROGRAM_FILE)
    end
    return nothing
end

# Convert a parameter value to something that can be stored in a TOML file
toml_value(value::Union{Bool, Integer, AbstractFloat, AbstractString}) = value
toml_value(value::Symbol) = string(value)
toml_value(value::Union{Tuple, AbstractVector}) = [toml_value(v) for v in value]
toml_value(value) = string(value)

# Characters that are safe in directory names on all operating systems
sanitize(text) = replace(string(text), r"[^A-Za-z0-9.+-]" => "-")

format_value(value::AbstractVector) = join(format_value.(value), ",")
format_value(value) = sanitize(value)

function run_label(params, name, source_file)
    name !== nothing && return sanitize(name)
    if isempty(params)
        # without parameters, interactive runs must not overwrite each other
        return source_file === nothing ?
               "run_" * Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS") : "default"
    end
    return join((key * "=" * format_value(value) for (key, value) in params), "__")
end

function prepare_run_directory!(run_info, semi)
    dir = run_info.dir
    if mpi_isroot()
        # Replace the results of a previous run with the same parameters, but never
        # delete a directory that was not created by `RunInfo`
        if isdir(dir)
            if isfile(joinpath(dir, "run.toml"))
                rm(dir; recursive = true)
            elseif !isempty(readdir(dir))
                error("The output directory `$dir` exists but was not created by " *
                      "`RunInfo`, so it is not overwritten. Move it or pass a different " *
                      "`name` or `output_root` to `@RunInfo`.")
            end
        end
        mkpath(dir)
    end

    copied = String[]
    try_io("copying the elixir and additional files") do
        for file in (run_info.source === nothing ? run_info.files :
                     vcat(run_info.source, run_info.files))
            if isfile(file)
                mpi_isroot() && cp(file, joinpath(dir, basename(file)), force = true)
                push!(copied, basename(file))
            else
                @warn "`RunInfo`: file `$file` not found, it is not copied."
            end
        end
    end

    try_io("writing setup.txt") do
        if mpi_isroot()
            open(joinpath(dir, "setup.txt"), "w") do io
                io = IOContext(io, :displaysize => (40, 100))
                mesh, equations, solver, _ = mesh_equations_solver_cache(semi)
                for item in (semi, mesh, equations, solver)
                    show(io, MIME"text/plain"(), item)
                    println(io)
                end
            end
        end
    end

    run_info.toml = Dict{String, Any}("run" => Dict{String, Any}("elixir" => run_info.elixir,
                                                                 "label" => run_info.label,
                                                                 "directory" => abspath(dir),
                                                                 "status" => "created",
                                                                 "created" => timestamp()),
                                      "parameters" => Dict{String, Any}(run_info.parameters),
                                      "environment" => environment_info(),
                                      "files" => Dict{String, Any}("copied" => copied,
                                                                   "setup" => "setup.txt"))
    if run_info.source !== nothing
        run_info.toml["run"]["source"] = run_info.source
        run_info.toml["files"]["elixir"] = basename(run_info.source)
    end
    write_run_toml(run_info)
    return nothing
end

timestamp() = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")

function environment_info()
    info = Dict{String, Any}("julia" => string(VERSION),
                             "trixi" => string(pkgversion(Trixi)),
                             "aerotrixi" => string(pkgversion(@__MODULE__)),
                             "threads" => Threads.nthreads(),
                             "mpi_ranks" => mpi_nranks(),
                             "hostname" => gethostname(),
                             "working_directory" => pwd())
    commit = git_commit(pkgdir(@__MODULE__))
    commit !== nothing && (info["aerotrixi_git_commit"] = commit)
    return info
end

# The git commit of a package in development mode, or `nothing`
function git_commit(dir)
    (dir === nothing || !isdir(joinpath(dir, ".git"))) && return nothing
    try
        cmd = pipeline(`git -C $dir rev-parse HEAD`, stderr = devnull)
        return strip(read(cmd, String))
    catch
        return nothing
    end
end

# Metadata must never stop a simulation: I/O problems only give a warning
function try_io(f, what)
    try
        f()
    catch err
        err isa InterruptException && rethrow()
        @warn "`RunInfo`: $what failed" exception=err
    end
    return nothing
end

function write_run_toml(run_info)
    try_io("writing run.toml") do
        if mpi_isroot()
            open(joinpath(run_info.dir, "run.toml"), "w") do io
                println(io, "# Description of a simulation run, written by AeroTrixi.jl")
                TOML.print(io, run_info.toml; sorted = true)
            end
        end
    end
end

# The callback is never triggered during the time integration; it only acts at the
# start and at the end of the simulation
(run_info::RunInfo)(u, t, integrator) = false
(run_info::RunInfo)(integrator) = nothing

function initialize_run_info!(cb, u, t, integrator)
    run_info = cb.affect!
    run_info.start_time = time()
    run_info.toml["run"]["status"] = "running"
    run_info.toml["run"]["started"] = timestamp()
    write_run_toml(run_info)
    return nothing
end

function finalize_run_info!(cb, u, t, integrator)
    run_info = cb.affect!
    semi = integrator.p
    mesh, _, solver, cache = mesh_equations_solver_cache(semi)
    iter = integrator.stats.naccept

    toml = run_info.toml
    toml["run"]["status"] = isfinished(integrator) ? "finished" : "stopped"
    toml["run"]["ended"] = timestamp()
    toml["run"]["wall_time_s"] = round(time() - run_info.start_time, digits = 2)
    toml["result"] = Dict{String, Any}("final_time" => Float64(integrator.t),
                                       "timesteps" => iter,
                                       "n_elements" => nelementsglobal(mesh, solver,
                                                                       cache))

    if run_info.save_final_state
        try_io("saving the final solution") do
            restart_file = save_final_state(run_info, integrator.u, integrator.t,
                                            integrator.dt, iter, semi)
            toml["files"]["restart"] = basename(restart_file)
            toml["files"]["mesh"] = splitdir(mesh.current_filename)[2]
        end
    end
    write_run_toml(run_info)
    return nothing
end

# Save the mesh and the solution as restart files in the run directory. The mesh is
# always written, even if it was saved elsewhere before, since `load_run` expects it
# next to the restart file.
function save_final_state(run_info, u_ode, t, dt, iter, semi)
    mesh, _, _, _ = mesh_equations_solver_cache(semi)
    mesh.current_filename = save_mesh_file(mesh, run_info.dir, iter)
    mesh.unsaved_changes = false
    restart_callback = SaveRestartCallback(output_directory = run_info.dir).affect!
    save_restart_file(u_ode, t, dt, iter, semi, restart_callback)
    return joinpath(run_info.dir, @sprintf("restart_%09d.h5", iter))
end

function Base.show(io::IO, run_info::RunInfo)
    print(io, "RunInfo(\"", run_info.dir, "\")")
end

function Base.show(io::IO, ::MIME"text/plain", run_info::RunInfo)
    if get(io, :compact, false)
        show(io, run_info)
    else
        setup = Pair{String, Any}["elixir" => run_info.elixir,
                                  "output directory" => abspath(run_info.dir)]
        for (key, value) in run_info.parameters
            push!(setup, "│ " * key => value)
        end
        push!(setup, "save final state" => run_info.save_final_state ? "yes" : "no")
        summary_box(io, "RunInfo", setup)
    end
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:RunInfo})
    @nospecialize cb # reduce precompilation time
    show(io, cb.affect!)
end

function Base.show(io::IO, mime::MIME"text/plain", cb::DiscreteCallback{<:Any, <:RunInfo})
    @nospecialize cb # reduce precompilation time
    show(io, mime, cb.affect!)
end

"""
    find_runs(directory = "out"; status = "finished", parameters...)

Return the directories of all runs stored with [`@RunInfo`](@ref) below `directory`,
e.g., `"out"` or `"out/<name of the elixir>"`, sorted by name.

Only runs with the given `status` are returned (`"finished"` by default; `nothing`
returns all). Further keyword arguments select runs by their parameters, e.g.,
`find_runs("out"; polydeg = 3)`.
"""
function find_runs(directory = "out"; status = "finished", parameters...)
    runs = String[]
    isdir(directory) || return runs
    for (root, _, files) in walkdir(directory)
        "run.toml" in files || continue
        toml = try
            TOML.parsefile(joinpath(root, "run.toml"))
        catch
            continue
        end
        run_status = get(get(toml, "run", Dict()), "status", "")
        status === nothing || run_status == status || continue
        run_parameters = get(toml, "parameters", Dict())
        matches = all(pairs(parameters)) do (key, value)
            haskey(run_parameters, string(key)) &&
                isequal(run_parameters[string(key)], toml_value(value))
        end
        matches && push!(runs, root)
    end
    return sort!(runs)
end

"""
    load_run(directory)

Load the final solution of a run stored with [`@RunInfo`](@ref).

The semidiscretization is rebuilt from the copy of the elixir in `directory`: the
elixir is evaluated up to the definition of `semi`, with the stored mesh and the
parameters recorded in `run.toml`. No time integration is performed.

Returns a `NamedTuple` with the fields `semi`, `u_ode` (the solution vector),
`time`, `info` (the contents of `run.toml`), and `dir`, so that, e.g.,
`PlotData1D(run.u_ode, run.semi; curve)` can be used for postprocessing.
"""
function load_run(directory)
    # `include` resolves relative paths with respect to the file currently being
    # included, not the working directory, so make the path absolute first
    directory = abspath(directory)
    info = TOML.parsefile(joinpath(directory, "run.toml"))
    files = get(info, "files", Dict())
    haskey(files, "restart") ||
        error("The run in `$directory` has no saved final state; was it finished?")
    haskey(files, "elixir") ||
        error("The run in `$directory` was not started from an elixir file.")
    restart_file = joinpath(directory, files["restart"])
    elixir = joinpath(directory, files["elixir"])

    mesh = load_mesh(restart_file)
    # Numbers and flags are passed back to the elixir, so that parameters that were
    # changed with `trixi_include` are restored as well
    overrides = Pair{Symbol, Any}[Symbol(key) => value
                                  for (key, value) in get(info, "parameters", Dict())
                                  if value isa Union{Bool, Real}]

    # Evaluate the elixir only up to the definition of `semi`
    semi_defined = Ref(false)
    function stop_after_semi(expr)
        semi_defined[] && return nothing
        if expr isa Expr && expr.head === :(=) && expr.args[1] === :semi
            semi_defined[] = true
        end
        return expr
    end
    mod = Module(:AeroTrixiLoadRun)
    trixi_include(stop_after_semi, mod, elixir; enable_assignment_validation = false,
                  mesh = mesh, overrides...)
    semi_defined[] || error("The elixir `$elixir` does not define `semi`.")

    semi = Base.invokelatest(getfield, mod, :semi)
    u_ode = load_restart_file(semi, restart_file)
    return (; semi, u_ode, time = load_time(restart_file), info, dir = directory)
end
