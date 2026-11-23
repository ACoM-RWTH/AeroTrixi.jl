# AeroTrixi.jl

**AeroTrixi.jl** is a package for high-fidelity aerodynamic simulations built on top of
[Trixi.jl](https://github.com/trixi-framework/Trixi.jl). It adds pointwise surface
quantities for aerodynamic applications, thermodynamics and entropy-conservative fluxes for
multi-species flows of calorically imperfect gases, and a bridge to continuous finite
elements of [Gridap.jl](https://github.com/gridap/Gridap.jl) for hyperbolic systems coupled
to elliptic equations. In the future, we plan to add more features for nonideal and
rarefied gases, for instance.

## Features

- **Surface pressure and friction coefficients**: the [`SurfacePointwiseCallback`](@ref)
  computes and saves pointwise aerodynamic coefficients along boundaries, such as
  [`SurfacePressureCoefficient`](@ref) and [`SurfaceFrictionCoefficient`](@ref)
  (see [Surface quantities](api/surface_quantities.md)).
- **Thermodynamic data**: tabulated internal energies and specific heats with interpolation
  and temperature inversion routines for multi-species flows in thermal equilibrium via
  [`ThermoData1T`](@ref), with models for rotational and vibrational energies
  (see [Thermodynamics](api/thermodynamics.md)).
- **Entropy-conservative/stable fluxes for non-calorically perfect multi-species flows**:
  [`CompressibleEulerEquationsMs1T2D`](@ref) for 2D multi-species flows based on tabulated
  thermodynamic data, with the entropy-conservative flux [`flux_oblapenko_etal`](@ref) and
  its faster approximation [`flux_oblapenko_etal_taylor`](@ref); chemical reactions are
  possible via source terms (see [Multi-species equations](api/multispecies_equations.md)).
- **Trixi.jl-Gridap.jl bridge**: continuous finite element problems, assembled with
  Gridap.jl, solved on the adaptive `P4estMesh` of a DG simulation (2D, DG with
  `polydeg = 2` and Q2 elements), with sources from and results written into the DG
  solution, on each Runge-Kutta stage via the [`GridapCallback`](@ref); for example, the
  Euler equations of a charged gas, [`CompressibleEulerPoissonEquations2D`](@ref) (see the
  [guide](trixi_gridap_bridge.md) and the [API](api/trixi_gridap_bridge.md)).

## Installation

AeroTrixi.jl is registered in the Julia General Registry. Install it using:

```julia
using Pkg
Pkg.add("AeroTrixi")
```

## Examples

To get started, it is best to take a look at the examples in the folder returned by
[`examples_dir`](@ref), which run with `trixi_include` like the examples of Trixi.jl:

| Feature | Examples |
|---|---|
| Aerodynamic flows and surface quantities | `p4est_2d_dgsem/elixir_euler_NACA0012airfoil_mach08.jl`, `p4est_2d_dgsem/elixir_navierstokes_NACA0012airfoil_mach08.jl`, `p4est_3d_dgsem/elixir_euler_ONERA_M6_AVI.jl` (with surface coefficients), `p4est_3d_dgsem/elixir_euler_DPW3_wing.jl` |
| Multi-species flows | `tree_2d_dgsem/elixir_eulermst1T_ec.jl` (entropy conservation), `tree_2d_dgsem/elixir_eulermst1T_chem.jl` (dissociation), `p4est_2d_dgsem/elixir_eulerms1T_1speciescylinder.jl`, `p4est_2d_dgsem/elixir_eulerms1T_2speciescylinder_chem.jl` (hypersonic cylinder) |
| Trixi.jl-Gridap.jl bridge | `p4est_2d_dgsem/elixir_euler_poisson_blast_five_trees_amr.jl` |

## Documentation

The [guide to the Trixi.jl-Gridap.jl bridge](trixi_gridap_bridge.md) explains its
foundations and usage. The [API Reference](api.md) documents all functions and types,
grouped by topic.

## Credit

### Referencing

If you use `AeroTrixi.jl` in your research, you should cite the upstream `Trixi.jl` repository:

```bibtex
@article{ranocha2022adaptive,
  title={Adaptive numerical simulations with {T}rixi.jl:
         {A} case study of {J}ulia for scientific computing},
  author={Ranocha, Hendrik and Schlottke-Lakemper, Michael and Winters, Andrew Ross
          and Faulhaber, Erik and Chan, Jesse and Gassner, Gregor},
  journal={Proceedings of the JuliaCon Conferences},
  volume={1},
  number={1},
  pages={77},
  year={2022},
  doi={10.21105/jcon.00077},
  eprint={2108.06476},
  eprinttype={arXiv},
  eprintclass={cs.MS}
}

@article{schlottkelakemper2021purely,
  title={A purely hyperbolic discontinuous {G}alerkin approach for
         self-gravitating gas dynamics},
  author={Schlottke-Lakemper, Michael and Winters, Andrew R and
          Ranocha, Hendrik and Gassner, Gregor J},
  journal={Journal of Computational Physics},
  pages={110467},
  year={2021},
  month={06},
  volume={442},
  publisher={Elsevier},
  doi={10.1016/j.jcp.2021.110467},
  eprint={2008.10593},
  eprinttype={arXiv},
  eprintclass={math.NA}
}
```

In addition, you can also refer to Trixi.jl directly as
```bibtex
@misc{schlottkelakemper2025trixi,
  title={{T}rixi.jl: {A}daptive high-order numerical simulations
         of hyperbolic {PDE}s in {J}ulia},
  author={Schlottke-Lakemper, Michael and Gassner, Gregor J and
          Ranocha, Hendrik and Winters, Andrew R and Chan, Jesse
          and Rueda-Ramírez, Andrés},
  year={2025},
  howpublished={\url{https://github.com/trixi-framework/Trixi.jl}},
  doi={10.5281/zenodo.3996439}
}
```

If using the entropy-conservative multi-species fluxes provided with `CompressibleEulerEquationsMs1T2D`, please also cite
```bibtex
@article{oblapenko2025entropyconservative,
  title={Entropy-conservative high-order methods for high-enthalpy gas flows},
  author={Oblapenko, Georgii and Torrilhon, Manuel},
  journal={Computers \& Fluids},
  volume={295},
  pages={106640},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.compfluid.2025.106640}
}

@inproceedings{oblapenko2024entropy,
  title={Entropy-stable fluxes for high-order Discontinuous Galerkin simulations of high-enthalpy flows},
  author={Oblapenko, Georgii and Tarnovskiy, Arseniy and Ertl, Moritz and Torrilhon, Manuel},
  booktitle={STAB/DGLR Symposium 2024},
  pages={393--402},
  year={2026},
  organization={Springer},
  doi={10.1007/978-3-032-11115-9_36}
}
```

### Acknowledgements

Furthermore, a lot of repository logistics and their structure (GitHub CI, testing, docs, etc.) are taken straight from `Trixi.jl`.
Thus, we greatly acknowledge all efforts from the [Trixi.jl Authors](https://github.com/trixi-framework/Trixi.jl/blob/main/AUTHORS.md).

We also thank Arpit Babbar who provided the [first implementation](https://github.com/trixi-framework/Trixi.jl/pull/1920) of point-wise analysis quantities.