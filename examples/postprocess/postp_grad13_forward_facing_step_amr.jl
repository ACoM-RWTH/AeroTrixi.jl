# postp_grad13_forward_facing_step_amr.jl
# ----------------------------------------------------------------------------
# Contour plot of the Grad13 forward-facing step solution of
# ../p4est_2d_dgsem/elixir_grad13_forward_facing_step_amr.jl, in the layout of the
# finite-volume solver's tools/explore_ffs.jl (MultiBlockMomentsFV): a wide 3:1 domain with
# the step cut out, a colourbar no taller than the axis, the step filled in light red, and the
# title placed inside the step as a white-boxed label.
#
# Usage in the REPL, after running the elixir (which defines `sol` and `semi`):
#
#   julia> include("examples/p4est_2d_dgsem/elixir_grad13_forward_facing_step_amr.jl")
#   julia> include("examples/postprocess/postp_grad13_forward_facing_step_amr.jl")
#
# or put the second `include` at the end of a copy of the elixir. Edit the settings below and
# `include` again to change the plot. Requires CairoMakie in the active environment (it is not a
# dependency of AeroTrixi.jl).
#
# The solution is sampled on a regular grid of `nx` × `nx/3` points by evaluating the DG
# polynomials of the (axis-aligned) elements; the plot variable is a name of the output
# variables of `cons2moments` (as written by the finite-volume solver) or a function of a
# NamedTuple of all of them plus the non-equilibrium measure `neq`.
# ----------------------------------------------------------------------------

import CairoMakie # all Makie calls are qualified so that the script also works with Plots loaded
using Trixi, AeroTrixi

# --- derived quantities (functions of the variables) ------------------------
# Each is called per grid point with a NamedTuple of ALL variables (nt.rho, nt.v1, nt.v2,
# nt.theta, nt.p, nt.P11, nt.P12, nt.P22, nt.q1, nt.q2, nt.neq). Pass any of these as `var`
# below together with a matching `label`.
speed = nt -> sqrt(nt.v1^2 + nt.v2^2)                 # |v|
mach = nt -> sqrt(nt.v1^2 + nt.v2^2) / sqrt(nt.theta) # Mach number (a = √θ)
# von Mises stress of the deviatoric pressure tensor, with P33 = 3p - P11 - P22
vonMises = nt -> sqrt(1.5 * ((nt.P11 - nt.p)^2 + (nt.P22 - nt.p)^2 +
                       (2 * nt.p - nt.P11 - nt.P22)^2 + 2 * nt.P12^2))
heatflux = nt -> sqrt(nt.q1^2 + nt.q2^2)                    # |q|

# --- plot settings ------------------------------------------------------------
var = "rho"        # a variable name, OR a function above (then set `label`)
label = "density"    # colourbar/title text (required when `var` is a function)
kind = :contourf    # :contourf or :heatmap
levels = 40           # number of contour levels
colormap = :viridis
nx = 600          # sampling resolution in x (ny = nx / 3)
mesh_lines = :black       # colour of the element outlines, or `nothing` for none
mesh_alpha = 0.35         # transparency of the element outlines
neq_levels = [0.1, 0.3]   # contour lines of the non-equilibrium measure, or `nothing`
neq_color = :white
outfile = nothing      # e.g. "ffs_density.png", or `nothing` to only display

# ----------------------------------------------------------------------------
# sample_dg: evaluate the DG solution of the axis-aligned elements on a regular grid of cell
# centres; returns xs, ys and an nx × ny matrix of NamedTuples with all variables (NaN inside
# the step, i.e., where no element exists).
# ----------------------------------------------------------------------------
function sample_dg(sol, semi; nx = 600)
    u = Trixi.wrap_array(sol.u[end], semi)
    dg, equations, cache = semi.solver, semi.equations, semi.cache
    nn = Trixi.nnodes(dg)
    nodes = dg.basis.nodes
    wbary = Trixi.barycentric_weights(nodes)
    xc = cache.elements.node_coordinates
    xmin, xmax = extrema(@view xc[1, :, :, :])
    ymin, ymax = extrema(@view xc[2, :, :, :])
    ny = round(Int, nx * (ymax - ymin) / (xmax - xmin))
    xs = collect(range(xmin, xmax, length = nx + 1))
    xs = 0.5 .* (xs[1:(end - 1)] .+ xs[2:end])
    ys = collect(range(ymin, ymax, length = ny + 1))
    ys = 0.5 .* (ys[1:(end - 1)] .+ ys[2:end])
    names = (:rho, :v1, :v2, :theta, :p, :P11, :P12, :P22, :q1, :q2, :neq)
    nan = NamedTuple{names}(ntuple(_ -> NaN, length(names)))
    Z = fill(nan, nx, ny)
    for e in 1:Trixi.nelements(dg, cache)
        x0, x1 = xc[1, 1, 1, e], xc[1, nn, 1, e]
        y0, y1 = xc[2, 1, 1, e], xc[2, 1, nn, e]
        i0, i1 = searchsortedfirst(xs, x0), searchsortedlast(xs, x1)
        j0, j1 = searchsortedfirst(ys, y0), searchsortedlast(ys, y1)
        for i in i0:i1
            ξ = -1 + 2 * (xs[i] - x0) / (x1 - x0)
            Lξ = Trixi.lagrange_interpolating_polynomials(ξ, nodes, wbary)
            for j in j0:j1
                η = -1 + 2 * (ys[j] - y0) / (y1 - y0)
                Lη = Trixi.lagrange_interpolating_polynomials(η, nodes, wbary)
                un = sum(Lξ[a] * Lη[b] * Trixi.get_node_vars(u, equations, dg, a, b, e)
                         for a in 1:nn, b in 1:nn)
                m = cons2moments(un, equations)
                Z[i, j] = NamedTuple{names}((m..., nonequilibrium(un, equations)))
            end
        end
    end
    return xs, ys, Z
end

# element outlines as line segments (one `linesegments!` call)
function element_outlines(semi)
    xc = semi.cache.elements.node_coordinates
    nn = Trixi.nnodes(semi.solver)
    segs = CairoMakie.Point2f[]
    for e in 1:Trixi.nelements(semi.solver, semi.cache)
        c = (CairoMakie.Point2f(xc[1, 1, 1, e], xc[2, 1, 1, e]),
             CairoMakie.Point2f(xc[1, nn, 1, e], xc[2, nn, 1, e]),
             CairoMakie.Point2f(xc[1, nn, nn, e], xc[2, nn, nn, e]),
             CairoMakie.Point2f(xc[1, 1, nn, e], xc[2, 1, nn, e]))
        push!(segs, c[1], c[2], c[2], c[3], c[3], c[4], c[4], c[1])
    end
    return segs
end

# ----------------------------------------------------------------------------
# plot_ffs_dg: the contour plot in the layout of the finite-volume script. `var` is a variable
# name or a function of the NamedTuple of variables; the step (the part of the bounding box
# without elements) is drawn in light red; `mesh_lines` draws the element outlines in that
# colour; `neq_levels` adds contour lines of the non-equilibrium measure.
# ----------------------------------------------------------------------------
function plot_ffs_dg(sol, semi, var; label = nothing, kind = :contourf, levels = 40,
                     colormap = :viridis, nx = 600, mesh_lines = nothing, mesh_alpha = 0.35,
                     neq_levels = nothing, neq_color = :white)
    xs, ys, Z = sample_dg(sol, semi; nx = nx)
    f = var isa Function ? var : (nt -> getfield(nt, Symbol(var)))
    V = [isnan(nt.rho) ? NaN : f(nt) for nt in Z]
    finite = filter(isfinite, V)
    colorrange = isempty(finite) ? (0.0, 1.0) : (minimum(finite), maximum(finite))

    xmin, xmax = xs[1] - 0.5 * (xs[2] - xs[1]), xs[end] + 0.5 * (xs[2] - xs[1])
    ymin, ymax = ys[1] - 0.5 * (ys[2] - ys[1]), ys[end] + 0.5 * (ys[2] - ys[1])
    # the step: lower-right rectangle without elements, from the first NaN column at the bottom
    jbot = 1
    ibot = findfirst(i -> isnan(Z[i, jbot].rho), eachindex(xs))
    sx0 = ibot === nothing ? xmax : xs[ibot] - 0.5 * (xs[2] - xs[1])
    jtop = ibot === nothing ? 1 : findfirst(j -> !isnan(Z[end, j].rho), eachindex(ys))
    sy1 = jtop === nothing ? ymin : ys[jtop] - 0.5 * (ys[2] - ys[1])
    sx1, sy0 = xmax, ymin

    aspect = (xmax - xmin) / (ymax - ymin)
    axh = 300
    fig = CairoMakie.Figure(size = (round(Int, axh * aspect) + 140, axh + 70))
    ax = CairoMakie.Axis(fig[1, 1]; aspect = CairoMakie.DataAspect(), xgridvisible = false,
                         ygridvisible = false)

    plt = if kind === :contourf
        lev = range(colorrange[1] - 1e-6, colorrange[2] + 1e-6; length = levels + 1)
        CairoMakie.contourf!(ax, xs, ys, V; levels = lev, colormap = colormap)
    elseif kind === :heatmap
        CairoMakie.heatmap!(ax, xs, ys, V; colormap = colormap, colorrange = colorrange,
                            nan_color = :transparent)
    else
        error("unknown kind=$kind")
    end

    if neq_levels !== nothing
        N = [isnan(nt.rho) ? NaN : nt.neq for nt in Z]
        CairoMakie.contour!(ax, xs, ys, N; levels = neq_levels, color = neq_color,
                            linewidth = 1.0)
    end
    if mesh_lines !== nothing
        CairoMakie.linesegments!(ax, element_outlines(semi);
                                 color = (mesh_lines, mesh_alpha),
                                 linewidth = 0.4)
    end

    # the step in light red, drawn on top so the blank cells read as a solid region
    CairoMakie.poly!(ax, CairoMakie.Point2f[(sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1)];
                     color = (:red, 0.22), strokecolor = (:firebrick, 0.6),
                     strokewidth = 0.8)
    CairoMakie.limits!(ax, xmin, xmax, ymin, ymax)

    vname = label !== nothing ? String(label) :
            var isa Function ?
            error("plot_ffs_dg: `var` is a function -- pass `label=...`") :
            String(var)
    CairoMakie.Colorbar(fig[1, 2], plt; width = 12, ticklabelsize = 14, labelsize = 14)
    CairoMakie.rowsize!(fig.layout, 1, CairoMakie.Aspect(1, 1 / aspect))
    CairoMakie.colgap!(fig.layout, 8)

    # title inside the step: a white box with a thin black frame in the bottom-right corner
    nel = Trixi.nelements(semi.solver, semi.cache)
    t = round(sol.t[end]; digits = 4)
    ttl = "G13 Moment Eqns, $(vname),  t = $t,  Kn = $(semi.equations.Kn)\nDG/AMR, $nel elements"
    pad = 0.012 * (xmax - xmin)
    bw = 0.95 * (sx1 - sx0)
    bh = 0.40 * (sy1 - sy0)
    bx1 = sx1 - pad
    bx0 = bx1 - bw
    by0 = sy0 + pad
    by1 = by0 + bh
    CairoMakie.poly!(ax, CairoMakie.Point2f[(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)];
                     color = :white, strokecolor = :black, strokewidth = 0.8)
    CairoMakie.text!(ax, 0.5 * (bx0 + bx1), 0.5 * (by0 + by1); text = ttl,
                     align = (:center, :center), fontsize = 16, color = :black,
                     space = :data)

    return fig, ax
end

fig, ax = plot_ffs_dg(sol, semi, var; label = label, kind = kind, levels = levels,
                      colormap = colormap, nx = nx, mesh_lines = mesh_lines,
                      mesh_alpha = mesh_alpha, neq_levels = neq_levels,
                      neq_color = neq_color)
display(fig)
outfile === nothing || CairoMakie.save(outfile, fig)

# ----------------------------------------------------------------------------
# Further plots from the REPL, e.g.:
#   fig, _ = plot_ffs_dg(sol, semi, vonMises; label = "von-Mises-stress", colormap = :deep); display(fig)
#   fig, _ = plot_ffs_dg(sol, semi, "theta"; label = "temperature", mesh_lines = :white)
#   fig, _ = plot_ffs_dg(sol, semi, nt -> nt.neq; label = "non-equilibrium", kind = :heatmap,
#                        neq_levels = [0.1, 0.3])
# ----------------------------------------------------------------------------
