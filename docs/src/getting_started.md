# Getting started: supersonic flow around an airfoil

This tutorial is a first entry point to simulations with Trixi.jl and AeroTrixi.jl.
It runs a single example, a Mach 2 flow around a NACA 6412 airfoil, which shows
the complete workflow of a simulation: from a geometry file to the mesh, the
boundary and initial conditions, the discretization, the time integration and
finally a plot of the result.

The example is based on material for the lecture
*Numerical Methods for Partial Differential Equations* by Manuel Torrilhon, RWTH Aachen.

## Installation

1. Install Julia with the version manager [juliaup](https://github.com/JuliaLang/juliaup).
   On Linux and macOS, type in a terminal
   ```
   curl -fsSL https://install.julialang.org | sh
   ```
   On Windows, install Julia from the Microsoft Store, or type `winget install julia -s msstore`.
   Afterwards, start Julia by typing `julia`.

2. In the Julia REPL, install the packages used in the example:
   ```julia
   using Pkg
   Pkg.add(["AeroTrixi", "Trixi", "OrdinaryDiffEqSSPRK", "Gmsh", "Plots"])
   ```
   - [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) is the solver,
   - [OrdinaryDiffEqSSPRK.jl](https://github.com/SciML/OrdinaryDiffEq.jl) provides the time integration method,
   - [Gmsh.jl](https://github.com/JuliaFEM/Gmsh.jl) generates the mesh; it contains
     the mesh generator [Gmsh](https://gmsh.info/), so there is nothing else to install,
   - [Plots.jl](https://github.com/JuliaPlots/Plots.jl) is used for the visualization.

The first installation and the first run take a few minutes, since Julia
compiles the packages.

## Running the example

The example consists of two files in the folder `examples/p4est_2d_dgsem` of AeroTrixi.jl:

- `elixir_euler_NACA6412airfoil_supersonic.jl`: the Julia script that sets up and runs the simulation,
- `NACA6412airfoil.geo`: the Gmsh geometry file of the airfoil in a box, constructed
  with the help of [GMSH-Airfoil-2D](https://github.com/cfsengineering/GMSH-Airfoil-2D).

Copy both files into a folder of your choice, so that you can change them:
```julia
import AeroTrixi
folder = joinpath(AeroTrixi.examples_dir(), "p4est_2d_dgsem")
cp(joinpath(folder, "elixir_euler_NACA6412airfoil_supersonic.jl"),
   "elixir_euler_NACA6412airfoil_supersonic.jl")
cp(joinpath(folder, "NACA6412airfoil.geo"), "NACA6412airfoil.geo")
```
Here `import` is used rather than `using`: both Trixi.jl and AeroTrixi.jl currently
provide a function called `AnalysisCallback`, and the example uses the one of Trixi.jl.

Then run the example with
```julia
include("elixir_euler_NACA6412airfoil_supersonic.jl")
```
Every 100 time steps a short summary of the solution is printed. There is no exact
solution for this flow, so no errors are computed.
At the end, the density together with the mesh is plotted and saved as `out/density.png`.
Other quantities can be plotted in the REPL with, e.g., `plot(pd["v1"])` or `plot(pd["p"])`.

The solution can also be written to files for [ParaView](https://www.paraview.org/)
or [VisIt](https://visit-dav.github.io/visit-website/); see the
[visualization section](https://trixi-framework.github.io/TrixiDocumentation/stable/visualization/)
of the Trixi.jl documentation.

## What the example does

The script is divided into six parts, which you find as sections in the file:

1. **Equations and initial condition**: the compressible Euler equations with
   ``\gamma = 1.4``. The whole domain is initially filled with the freestream state
   ``\rho = 1.4``, ``v_1 = 2``, ``v_2 = 0``, ``p = 1``, which has a speed of sound of 1,
   i.e. the flow has Mach number 2.
2. **Boundary conditions**: at the supersonic inflow (left) the flux is computed from
   the freestream state, at the supersonic outflow (right) from the state inside the
   domain. The airfoil and the top and bottom of the box are slip walls.
3. **Mesh**: Gmsh generates a mesh of quadrilaterals from the `.geo` file. The
   parameter `mesh_size` controls the size of the cells. The boundaries are the
   named `Physical Line`s of the `.geo` file: `inflow`, `outflow`, `airfoil` and `walls`.
4. **Spatial discretization**: the discontinuous Galerkin spectral element method
   (DGSEM) with polynomials of degree `polydeg` in each cell. Near shocks the method
   is blended with a robust finite volume method, which is controlled by a shock indicator.
5. **Time integration**: a strong stability preserving Runge-Kutta method up to
   ``t = 0.6``, with the time step chosen by the CFL condition and a limiter that
   keeps density and pressure positive.
6. **Visualization**: a plot of the density and the mesh.

## Experimenting with the resolution

The resolution of the simulation is controlled by two parameters in the script:
the polynomial degree `polydeg` and the relative mesh size `mesh_size`.
Instead of editing the file, you can also change them when running it:
```julia
using Trixi
trixi_include("elixir_euler_NACA6412airfoil_supersonic.jl", polydeg = 1, mesh_size = 0.05)
```

Try, for instance, the following four combinations and compare the density at ``t = 0.6``:

| `polydeg` | `mesh_size` | what to expect |
|:---------:|:-----------:|:---------------|
| 1 | 0.1  | a relatively coarse and diffusive representation of the shock waves evolving from the airfoil |
| 3 | 0.1  | the higher polynomial degree in each cell sharpens the shock waves |
| 1 | 0.05 | alternatively, the mesh can be refined while keeping the polynomial degree low, which gives an overall comparable resolution |
| 3 | 0.05 | a finer mesh together with a higher polynomial degree takes the longest time to compute, but gives the sharpest result |

The default setting, `polydeg = 2` and `mesh_size = 0.1`, is chosen to run fast
while still giving a reasonable result.
