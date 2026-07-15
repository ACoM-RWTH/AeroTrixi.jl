# AeroTrixi.jl

**AeroTrixi.jl** is a package for high-fidelity aerodynamic simulations built on top of [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).
Currently, it extends Trixi.jl's capabilities with a specialized analysis callback for aerodynamic applications.
In the future, we plan to add more features for nonideal and rarefied gases, for instance.

## Features

- **Surface pressure and friction coefficients**: Compute and save pointwise aerodynamic coefficients along boundaries

## Installation

AeroTrixi.jl is registered in the Julia General Registry. Install it using:

```julia
using Pkg
Pkg.add("AeroTrixi")
```

## Quick Start

To get started, it is best to take a look at the examples.
Currently, the only additional functionality by AeroTrixi.jl is the extended
`AnalysisCallback` which can be used to compute pointwise aerodynamic coefficients along boundaries,
such as `SurfacePressureCoefficient` and `SurfaceFrictionCoefficient`.

## Documentation

See the [API Reference](@ref) for documentation on available/extended functions.


## Credit

### Referencing

If you use `AeroTrixi.jl` in your research, you should cite the upstream `Trixi.jl` repository

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

### Acknowledgements

Furthermore, a lot of repository logistics and their structure (GitHub CI, testing, docs, etc.) are taken straight from `Trixi.jl`.
Thus, we greatly acknowledge all efforts from the [Trixi.jl Authors](https://github.com/trixi-framework/Trixi.jl/blob/main/AUTHORS.md).

We also thank Arpit Babbar who provided the [first implementation](https://github.com/trixi-framework/Trixi.jl/pull/1920) of point-wise analysis quantities.