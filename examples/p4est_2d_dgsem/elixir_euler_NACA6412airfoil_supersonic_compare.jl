using Trixi
using AeroTrixi: find_runs, load_run
using Plots

###############################################################################
# Comparison of the runs of `elixir_euler_NACA6412airfoil_supersonic_amr.jl`.
#
# Run `elixir_euler_NACA6412airfoil_supersonic_amr.jl` first for one or more setups,
# e.g., with and without AMR. This file then collects all finished runs stored in
# `out/elixir_euler_NACA6412airfoil_supersonic_amr/`, extracts the density and the
# pressure along a line in front of the airfoil, and plots them together.

###############################################################################
# 1. collect the stored runs

runs_directory = joinpath("out", "elixir_euler_NACA6412airfoil_supersonic_amr")

# All finished runs; select runs with keywords, e.g., `find_runs(runs_directory; use_amr = true)`
runs = find_runs(runs_directory)
isempty(runs) &&
    error("No finished runs found in `$runs_directory`. Run " *
          "`elixir_euler_NACA6412airfoil_supersonic_amr.jl` first.")

###############################################################################
# 2. the line: from x = -0.6 to the leading edge of the airfoil, at its height

x_leading_edge = -0.4900332889206208 # point 5 of the .geo file
y_leading_edge = 0.09933466539753061
x_start = -0.6
x_line = range(x_start, x_leading_edge - 1.0e-4, length = 400)
curve = vcat(x_line', fill(y_leading_edge, 1, length(x_line)))

###############################################################################
# 3. extract the solution along the line for each run and plot it

# a readable label from the parameters and results stored in `run.toml`
function run_label(info)
    parameters = info["parameters"]
    setup = parameters["use_amr"] ?
            "AMR, max_level $(parameters["max_level"])" :
            "static, mesh_size $(parameters["mesh_size"])"
    result = info["result"]
    return "$setup: $(result["n_elements"]) cells, $(info["run"]["wall_time_s"]) s"
end

# colors in a fixed order; AMR runs are drawn with solid lines, static meshes dashed
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
    "#e34948"]
length(runs) > length(colors) &&
    @warn "Only the first $(length(colors)) of $(length(runs)) runs are plotted."

plot_density = plot(xlabel = "x", ylabel = "density", legend = :topleft)
plot_pressure = plot(xlabel = "x", ylabel = "pressure", legend = :topleft)

for (color, directory) in zip(colors, runs)
    stored_run = load_run(directory)
    # `PlotData1D` interpolates the solution to the points of the curve
    line = PlotData1D(stored_run.u_ode, stored_run.semi; curve,
                      solution_variables = cons2prim)
    x = x_start .+ line.x # `line.x` is the arc length along the curve
    style = stored_run.info["parameters"]["use_amr"] ? :solid : :dash
    label = run_label(stored_run.info)
    plot!(plot_density, x, line.data[:, 1]; label, color, linestyle = style,
          linewidth = 2)
    plot!(plot_pressure, x, line.data[:, 4]; label, color, linestyle = style,
          linewidth = 2)
end

plot(plot_density, plot_pressure, layout = (2, 1), size = (1100, 900))
savefig(joinpath(runs_directory, "comparison.png"))
