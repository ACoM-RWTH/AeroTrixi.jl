# AUSM+ flux

[`flux_ausmplus`](@ref) is the AUSM+ flux for Trixi.jl's `CompressibleEulerEquations1D`,
which can be used as surface flux or in finite volume schemes.
The AUSM+-up [`FluxAUSMPlusUp`](@ref) of Liou (2006), an extension of [`flux_ausmplus`](@ref) to all speeds,
is also available (default instance is `flux_ausmplusup`).

```@autodocs
Modules = [AeroTrixi]
Pages = ["equations/compressible_euler_1d.jl"]
```
