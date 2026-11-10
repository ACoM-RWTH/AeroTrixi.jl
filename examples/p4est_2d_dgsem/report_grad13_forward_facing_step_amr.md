# Grad13 forward-facing step in AeroTrixi.jl: session report

Session of 2026-11-10. Repositories: `ACoM-RWTH/AeroTrixi.jl` (branches `claude/ffs-treemesh` and `claude/ffs-gmsh`,
the latter renamed from `claude/focused-rubin-ijzhkk` at the end of the session) and
`torrilhon/MultiBlockMomentsFV` (branch `claude/focused-rubin-ijzhkk`, merged as PR #4).

## 1. Goal

Reproduce the supersonic forward-facing step (FFS) simulation of Grad's 13-moment equations (Grad13)
from the C finite-volume code `MultiBlockMomentsFV` in `AeroTrixi.jl`, with a discontinuous Galerkin
(DG) discretization on a `P4estMesh` and adaptive mesh refinement (AMR), and provide a Trixi-side
post-processing that draws the same figures as the FV tools.

## 2. Analysis of the FV code and the plan

The FV code solves the 2D Grad13 system with 9 conservative variables (density, momentum, energy,
three stress-type moments, two heat-flux moments) on multi-block structured grids: MUSCL
reconstruction with limiters (none, average, minmod, superbee, harmonic), an HLL flux with clipped
wave speeds, Heun time stepping, explicit relaxation of stress and heat flux with the collision
frequency ν = ρ√θ/Kn, specular walls via ghost cells with flipped odd components, Dirichlet inflow,
and binary output read by `tools/load_solution.jl` and plotted by `tools/explore_ffs.jl` (CairoMakie).
The FFS case `ffs28.txt` at refinement 6 has 302400 cells (h = 1/360).

Decisions taken in discussion:

- Polynomial degree 2 (degree 1 makes the modal shock indicators degenerate).
- Collision terms as explicit source terms; the stiffness ν Δt is handled by the CFL number only.
- AMR driven by a single non-equilibrium measure combining stress and heat flux.
- Specular walls with accommodation zero for general wall normals (odd components flipped).
- Shock capturing first with the Hennemann–Gassner (HG) indicator, later replaced (Section 5).

## 3. Implementation in AeroTrixi.jl

`src/equations/grad13_2d.jl` adds `Grad13Equations2D` with the Grad closure, conversions
`cons2prim`, `prim2cons`, `cons2moments` (the FV output set), the Cartesian and normal-direction
fluxes, tensor rotations, the FV wave-speed bound 2√(5θ/3), a clipped HLL flux `flux_hll_grad13`,
the explicit source term `source_terms_grad13`, the specular wall `boundary_condition_slip_wall`
(P4est and TreeMesh versions), the diagnostics `von_mises_stress`, `nonequilibrium`
(Q = sqrt((σ_vM/p)² + |q|²/(p² θ))), `collision_frequency`, and the analysis quantity
`max_collision_frequency`.

`src/meshes/p4est_mesh_forward_facing_step.jl` tiles the channel [0,3]×[0,1] minus the step with 21
square trees and names the boundaries `inflow`, `outflow`, `wall`.

Examples: `examples/p4est_2d_dgsem/elixir_grad13_forward_facing_step_amr.jl` (the FFS case) and
`examples/tree_2d_dgsem/elixir_grad13_riemann2d.jl` (the FV test case `square4` for validation).
Tests: 74 unit tests against numbers generated from the C code, plus elixir tests for both examples
with positivity and allocation checks. README and docs were updated.

Validation: the Riemann profiles converge to the FV reference, and the FFS fields at t = 0.3 and
t = 1 match the FV figures at Kn = 0.005 and 0.001. Along the way, a bug in the FV output was found
and fixed in `src/Output.c` of MultiBlockMomentsFV (temperature, pressure and pressure-tensor columns
were written with the wrong normalization); the reference files were regenerated and merged into
`main` as PR #4 on 2026-11-08.

## 4. Mesh route: gmsh detour and return to the square-tree mesh

A gmsh route was built on the branch now called `claude/ffs-gmsh`: a `.geo` file with a global mesh size and
refinement nodes near the step, the Abaqus `.inp` read by Trixi, boundary conditions assigned by the
`PhysicalLine` labels, uniform p4est refinement of the loaded tree, and a non-adaptive run. The user
adjusted the geometry; the route works but was set aside ("we keep it for later").

Work continued on `claude/ffs-treemesh` with the self-made square-tree mesh. Requests implemented on
the way: removal of the `RelaxationStepsizeCallback` and of the `ControllerMinimumSize` helper, so
that the time step is controlled by `StepsizeCallback(cfl)` alone and the AMR by a plain
`ControllerThreeLevel`; an AMR configuration with at most 10000 elements at t = 1 (levels 2/4/6).

## 5. Robustness study and the final discretization

Observed failure modes and the measures that resolved them:

- Negative temperature at the step corner: Zhang–Shu positivity limiter on density and pressure
  (pressure is concave, temperature is not).
- The Trixi `FluxHLL` returns the left flux at the stationary sonic state at the step face, which
  the inflow hits exactly: the clipped HLL flux `flux_hll_grad13` with a central fallback.
- With HLL, a density spike grows at the head of the bow shock and at level interfaces: the
  Lax–Friedrichs flux with the FV wave-speed bound is used instead.
- AMR projection of a shock born at the step face produces children with negative mean pressure:
  cured by second-order subcell blending (below); static refinement was no longer needed.
- Pure DG without blending fails at Kn = 0.001 for degree 1 and 2, with and without a CFL ramp,
  on finer meshes and with HLL (overshoots and negative temperatures in the shock layers).
- `VolumeIntegralShockCapturingHG` (first-order subcells) is robust but smears the reflected shock.
  `VolumeIntegralShockCapturingRRG` (limited second-order subcells, slopes in primitive variables)
  with the van Leer limiter, which equals the FV "harmonic" limiter, keeps the FV sharpness.
  Minmod and the unlimited central slope were also tested.

Step count DG versus FV: the DG time step is bounded by h/(nnodes (λ₁+λ₂)) on level-6 elements of
size 1/192 with three nodes, which is a smaller step than the FV cell of 1/360 with Heun, hence
about 10500 DG steps versus 2200 FV steps to t = 1.

Blending indicator: analysis of the HG α per element versus the element maximum of Q showed that Q
itself is a better physical criterion. A 1D hyperbolicity scan gave the loss of real
characteristic speeds at Q ≈ 0.6–0.9 (heat flux 0.64 p√θ, compressive stress 0.8 p); per-element
statistics at Kn = 0.001 gave Q ≤ 0.03 in smooth flow, ≤ 0.25 in the expansion and the shear layer,
0.3–0.9 in the shock layers. The elixir therefore defines `IndicatorNonequilibriumBlending`, which maps
the element maximum of Q linearly onto [0, α_max] between `lo` and `hi` and applies the usual
neighbour smoothing. With lo = 0.1, hi = 0.3, α_max = 0.5, about 27% of the elements are blended
(47% with HG) at the same solution; the role of the three parameters was documented with an
α-over-Q plot.

Final configuration of the elixir (state at the end of the session, after the user's last commit):
degree 2, Lax–Friedrichs surface flux, central volume flux, RRG blending with van Leer,
non-equilibrium blending indicator (0.1 / 0.3 / 0.5), AMR levels 2/4/6 with thresholds 0.01 and
0.04 on `IndicatorMax(nonequilibrium)`, CFL 0.5, Zhang–Shu limiter, `CarpenterKennedy2N54`,
Kn = 0.005 (the user's final setting; the development and validation used 0.001).

## 6. Post-processing script

`examples/postprocess/postp_grad13_forward_facing_step_amr.jl` reproduces the layout of the FV
script `tools/explore_ffs.jl` from the Trixi result: after `include` of the elixir, `include` of the
script samples the DG polynomials on a regular grid (nx × nx/3, 600 by default) and draws a filled
contour or heatmap with the colourbar pinned to the axis height, the step in light red and the title
in a white box inside the step. The variable is a name of the FV output set or a function of all
variables (`speed`, `mach`, `vonMises`, `heatflux`, `neq` are predefined), with options for levels,
colormap, mesh lines in a chosen colour and transparency, contour lines of the non-equilibrium
measure at chosen levels, an output file, and a `show_figure` switch. All Makie calls are qualified
so the script also works with Plots loaded. The elixir contains no plot commands.

Two display-related commits (an external viewer with a pinned window position) were reverted at the
user's request; the user restored the simple `show_figure` switch and chose the final settings
(`:turbo` colormap, brown mesh lines, non-equilibrium level 0.05 in orange, output `plot.png`).

## 7. State of the branches

- `AeroTrixi.jl`, `claude/ffs-treemesh` (head ceabf6c, pushed): equations, mesh helper, both
  elixirs, tests, docs, post-processing script. This is the branch to merge.
- `AeroTrixi.jl`, `claude/ffs-gmsh` (head 958d81a, renamed from `claude/focused-rubin-ijzhkk`): the
  gmsh route (geo, inp, label-based boundary conditions), kept for later. All of its commits are
  also contained in `claude/ffs-treemesh`.
- `MultiBlockMomentsFV`, PR #4 (the `Output.c` fix and regenerated references) is merged into `main`;
  its branch `claude/focused-rubin-ijzhkk` has no unmerged commits. Note that the committed
  `Scenario.c` keeps the FFS case active, so `make check` does not run the Riemann references
  without switching the scenario.
- Branches still to delete on GitHub (the session's git access cannot delete remote branches):
  `claude/focused-rubin-ijzhkk` in both repositories, now superseded by `claude/ffs-gmsh` in
  AeroTrixi.jl and by the merged PR in MultiBlockMomentsFV.

Open points: the gmsh route is unfinished for AMR runs; the AMR thresholds and the blending
thresholds are absolute values of Q and scale with the Knudsen number, so other Kn need a check; the
elixir test values in `test/test_p4est.jl` correspond to Kn = 0.001 and the earlier thresholds and
must be regenerated for the user's final Kn = 0.005 settings before the test suite passes.
