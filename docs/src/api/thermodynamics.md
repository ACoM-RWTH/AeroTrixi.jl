# Thermodynamics

[`ThermoData1T`](@ref) holds tabulated internal energies and specific heats of
multi-species flows in thermal equilibrium, with interpolation and the inversion for the
temperature. The rotational and vibrational contributions come from the models below.

## Tabulated thermodynamic data

```@autodocs
Modules = [AeroTrixi]
Pages = ["thermo_models/thermo_models.jl", "thermo_models/thermo_interpolated.jl",
         "thermo_models/interpolation_types.jl"]
```

## Rotational and vibrational energies

```@autodocs
Modules = [AeroTrixi]
Pages = ["thermo_models/spectra.jl"]
```
