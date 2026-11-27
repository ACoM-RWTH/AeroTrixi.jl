# Visco-resistive MHD boundary conditions

Boundary conditions for the visco-resistive MHD equations of Trixi.jl, i.e.,
[`IdealGlmMhdEquations2D`](@extref Trixi.IdealGlmMhdEquations2D) for the hyperbolic part
and [`ViscoResistiveMhd2D`](@extref Trixi.ViscoResistiveMhd2D) for the parabolic part:
a slip wall with vanishing magnetic field ([`boundary_condition_slip_wall_no_magnetic_field`](@ref))
for the hyperbolic part and a wall with prescribed velocity, heat flux, and magnetic field
([`BoundaryConditionVRMHDWall`](@ref) with [`Isomagnetic`](@ref)) for the parabolic part.
At outflow boundaries, the parabolic fluxes can be taken from the inside with
`boundary_condition_do_nothing` of Trixi.jl. See the example
`p4est_2d_dgsem/elixir_vrmhd_vortex_street.jl`.

```@autodocs
Modules = [AeroTrixi]
Pages = ["equations/visco_resistive_mhd_2d.jl"]
```
