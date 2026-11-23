# Surface quantities

AeroTrixi.jl computes pointwise aerodynamic coefficients, such as the pressure and the skin
friction coefficient, along boundaries and saves them with the
[`SurfacePointwiseCallback`](@ref).

```@autodocs
Modules = [AeroTrixi]
Pages = ["callbacks_step/surface_pointwise_callback.jl",
         "callbacks_step/analysis_surface_pointwise.jl",
         "callbacks_step/analysis_surface_pointwise_2d.jl",
         "callbacks_step/analysis_surface_pointwise_3d.jl",
         "callbacks_step/analysis_surface_integral.jl",
         "callbacks_step/analysis_surface_integral_2d.jl",
         "callbacks_step/analysis_surface_integral_3d.jl"]
```
