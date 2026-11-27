# API Reference

The functions and types of AeroTrixi.jl, grouped by topic:

- [Surface quantities](api/surface_quantities.md): pointwise aerodynamic coefficients along
  boundaries
- [Thermodynamics](api/thermodynamics.md): tabulated thermodynamic data and models for
  rotational and vibrational energies
- [Multi-species equations](api/multispecies_equations.md): Euler equations for
  multi-species flows of calorically imperfect gases and their fluxes
- [AUSM+ flux](api/ausmplus.md): AUSM+ flux for the 1D compressible Euler equations
- [Visco-resistive MHD](api/visco_resistive_mhd.md): boundary conditions for the
  visco-resistive MHD equations
- [Trixi.jl-Gridap.jl bridge](api/trixi_gridap_bridge.md): continuous finite element
  problems on the adaptive mesh of a DG simulation (see also the
  [guide](trixi_gridap_bridge.md))

```@autodocs
Modules = [AeroTrixi]
Pages = ["src/AeroTrixi.jl", "auxiliary.jl"]
```

## Index

```@index
Pages = ["api/surface_quantities.md", "api/thermodynamics.md",
         "api/multispecies_equations.md", "api/ausmplus.md",
         "api/visco_resistive_mhd.md",
         "api/trixi_gridap_bridge.md", "api.md"]
```
