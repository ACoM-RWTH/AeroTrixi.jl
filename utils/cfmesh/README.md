# Boundary-layer airfoil meshes for `P4estMesh` with cfMesh

This directory contains a complete workflow to generate a 2D all-quadrilateral
mesh with boundary layers around the ONERA OAT 15A airfoil with
[cfMesh](https://cfmesh.com) and to convert it to an Abaqus `.inp` file that can
be read by Trixi.jl's `P4estMesh`. The boundaries are stored as node sets
`airfoil` and `farfield`. The workflow is explained on the documentation page
"Boundary-layer meshes with cfMesh" (`docs/src/cfmesh_meshes.md`) and is used
for the elixir `examples/p4est_2d_dgsem/elixir_navierstokes_OAT15Aairfoil.jl`.

Requirements: the cfMesh command-line tools `surfaceFeatureEdges` and
`cartesian2DMesh` (tested with CF-MESH+ 5.4), and Python 3 with NumPy and SciPy
(Matplotlib for the optional preview in `check_mesh.py`).

```bash
CFMESH_BIN=/path/to/cfMesh/bin ./Allmesh out_oat15a
```
creates `out_oat15a/oat15a_airfoil.inp`, which is used in Trixi.jl via
```julia
mesh = P4estMesh{2}("out_oat15a/oat15a_airfoil.inp",
                    boundary_symbols = [:airfoil, :farfield])
```

| File | Purpose |
|---|---|
| `Allmesh` | Driver script running all steps in a separate case directory |
| `oat15a_coordinates_c015.dat` | Airfoil coordinates (chord length 0.15, rescaled to 1) |
| `generate_geometry.py` | Airfoil spline and far-field circle as a "ribbon" STL (input for `cartesian2DMesh`) |
| `system/meshDict` | cfMesh settings: refinement and boundary layers of the coarse base mesh |
| `system/controlDict`, `fvSchemes`, `fvSolution` | Minimal dictionaries required by the OpenFOAM-based cfMesh tools |
| `polymesh_to_inp.py` | Converts the cfMesh (OpenFOAM `polyMesh`) mesh to conforming quads in Abaqus `.inp` format |
| `check_mesh.py` | Optional quality check of the cfMesh base mesh (substitute for OpenFOAM's `checkMesh`) |
| `inp_to_msh.py` | Converts a 2D Abaqus `.inp` mesh with node sets to the Gmsh format (MSH 2.2, ASCII) for inspection, e.g., `python3 inp_to_msh.py out_oat15a/oat15a_airfoil.inp` |

**Note:** the height of the first boundary layer cannot be prescribed with
cfMesh's command-line tools. It results from the cell size at the airfoil, the
number of layers, and their growth ratio (see `system/meshDict`); the value is
reported by `polymesh_to_inp.py`. The current settings give 10 layers with ratio
1.44 in the base mesh, i.e., 20 layers with ratio 1.2 and a first layer height
of about 2.5e-5 chord lengths in the final mesh.

cfMesh meshes contain polygonal cells with hanging nodes at the transitions
between refinement levels. `polymesh_to_inp.py` therefore splits every cell
into quadrilaterals, which halves all cell sizes; `system/meshDict` describes
the base mesh at twice the target cell sizes accordingly.
