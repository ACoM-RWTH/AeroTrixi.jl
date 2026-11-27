# Boundary conditions for the visco-resistive MHD equations of Trixi.jl,
# i.e., `IdealGlmMhdEquations2D` (hyperbolic part) and `ViscoResistiveMhd2D` (parabolic part).

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    boundary_condition_slip_wall_no_magnetic_field(u_inner, normal_direction::AbstractVector,
                                                   x, t, surface_flux_functions,
                                                   equations::IdealGlmMhdEquations2D)

Slip wall boundary condition for the hyperbolic part of the (visco-resistive) MHD
equations, which additionally imposes a vanishing magnetic field at the wall, e.g.,
for a non-magnetic body in a magnetized flow.
The boundary state is the mirrored inner state, i.e., the normal velocity is reflected,
while density, pressure, the out-of-plane velocity `v3`, and the divergence cleaning
field `psi` are copied, and all components of the magnetic field are set to zero.
The conservative and nonconservative surface fluxes are evaluated with the inner and
the boundary state.

Use this boundary condition for the hyperbolic part together with a
[`BoundaryConditionVRMHDWall`](@ref) with an [`Isomagnetic`](@ref) condition
`B = 0` for the parabolic part.
"""
@inline function boundary_condition_slip_wall_no_magnetic_field(u_inner,
                                                                normal_direction::AbstractVector,
                                                                x, t,
                                                                surface_flux_functions,
                                                                equations::IdealGlmMhdEquations2D)
    surface_flux_function, nonconservative_flux_function = surface_flux_functions

    # Normalize the vector without using `normalize` since we need to multiply by
    # the `norm_` later to obtain the correct scaling of the flux
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    rho, v1, v2, v3, p, _, _, _, psi = cons2prim(u_inner, equations)

    # Reflect the normal velocity and impose a vanishing magnetic field
    v_normal = normal[1] * v1 + normal[2] * v2
    B_wall = zero(rho)
    u_boundary = prim2cons(SVector(rho,
                                   v1 - 2 * v_normal * normal[1],
                                   v2 - 2 * v_normal * normal[2],
                                   v3, p, B_wall, B_wall, B_wall, psi), equations)

    flux = surface_flux_function(u_inner, u_boundary, normal, equations) * norm_
    noncons_flux = nonconservative_flux_function(u_inner, u_boundary, normal,
                                                 equations) * norm_
    return flux, noncons_flux
end

"""
    BoundaryConditionVRMHDWall(boundary_condition_velocity,
                               boundary_condition_heat_flux,
                               boundary_condition_magnetic)

Wall boundary condition for the parabolic part of the visco-resistive MHD equations
[`ViscoResistiveMhd2D`](@extref Trixi.ViscoResistiveMhd2D), analogous to
[`BoundaryConditionNavierStokesWall`](@extref Trixi.BoundaryConditionNavierStokesWall)
of the compressible Navier-Stokes equations. It combines

- a velocity boundary condition, currently [`NoSlip`](@extref Trixi.NoSlip),
- a heat flux boundary condition, currently [`Adiabatic`](@extref Trixi.Adiabatic),
- a magnetic field boundary condition, currently [`Isomagnetic`](@ref).

The combination `NoSlip`, `Adiabatic`, `Isomagnetic` is implemented for
`ViscoResistiveMhd2D` with `GradientVariablesPrimitive()`. With prescribed boundary
values of the velocity and the magnetic field, it can also be used at inflow
boundaries, see the example `p4est_2d_dgsem/elixir_vrmhd_vortex_street.jl`.
"""
struct BoundaryConditionVRMHDWall{V, H, B}
    boundary_condition_velocity::V
    boundary_condition_heat_flux::H
    boundary_condition_magnetic::B
end

"""
    Isomagnetic(boundary_value_function)

Prescribes the magnetic field `SVector(B1, B2, B3)` on a boundary of the visco-resistive
MHD equations via `boundary_value_function(x, t, equations)`, for use in a
[`BoundaryConditionVRMHDWall`](@ref).
"""
struct Isomagnetic{F}
    boundary_value_function::F # value of the magnetic field on the boundary
end

# Gradient step: the boundary values of the gradient variables. For
# `GradientVariablesPrimitive`, these are `(rho, v1, v2, v3, T, B1, B2, B3, psi)`, see
# `Trixi.cons2prim_temperature`. The velocity and the magnetic field are prescribed, all
# other variables are copied from the inner state (zero normal gradient of the
# temperature for an adiabatic wall).
@inline function (boundary_condition::BoundaryConditionVRMHDWall{<:NoSlip,
                                                                 <:Adiabatic,
                                                                 <:Isomagnetic})(flux_inner,
                                                                                 u_inner,
                                                                                 normal::AbstractVector,
                                                                                 x, t,
                                                                                 operator_type::Gradient,
                                                                                 equations::ViscoResistiveMhd2D{GradientVariablesPrimitive})
    v1, v2, v3 = boundary_condition.boundary_condition_velocity.boundary_value_function(x,
                                                                                        t,
                                                                                        equations)
    B1, B2, B3 = boundary_condition.boundary_condition_magnetic.boundary_value_function(x,
                                                                                        t,
                                                                                        equations)
    return SVector(u_inner[1], v1, v2, v3, u_inner[5], B1, B2, B3, u_inner[9])
end

# Divergence step: the normal parabolic fluxes. The momentum and induction fluxes are
# taken from the inside; the energy flux is formed with the prescribed boundary values of
# the velocity and the magnetic field and the prescribed normal heat flux, consistent with
# the energy flux of `ViscoResistiveMhd2D`, whose magnetic part is `B ⋅ (f6, f7, f8)`.
@inline function (boundary_condition::BoundaryConditionVRMHDWall{<:NoSlip,
                                                                 <:Adiabatic,
                                                                 <:Isomagnetic})(flux_inner,
                                                                                 u_inner,
                                                                                 normal::AbstractVector,
                                                                                 x, t,
                                                                                 operator_type::Divergence,
                                                                                 equations::ViscoResistiveMhd2D{GradientVariablesPrimitive})
    # `flux_inner` is the parabolic flux dotted with `normal`, which need not be a
    # unit vector (it includes the surface scaling on curvilinear meshes). Thus, the
    # prescribed normal heat flux must be scaled by `norm(normal)` as well.
    (; boundary_value_normal_flux_function) = boundary_condition.boundary_condition_heat_flux
    normal_heat_flux = boundary_value_normal_flux_function(x, t, equations) *
                       norm(normal)
    v1, v2, v3 = boundary_condition.boundary_condition_velocity.boundary_value_function(x,
                                                                                        t,
                                                                                        equations)
    B1, B2, B3 = boundary_condition.boundary_condition_magnetic.boundary_value_function(x,
                                                                                        t,
                                                                                        equations)

    _, tau_1n, tau_2n, tau_3n, _, Bvisc_1n, Bvisc_2n, Bvisc_3n, _ = flux_inner

    normal_energy_flux = (v1 * tau_1n + v2 * tau_2n + v3 * tau_3n + normal_heat_flux +
                          B1 * Bvisc_1n + B2 * Bvisc_2n + B3 * Bvisc_3n)

    return SVector(flux_inner[1], flux_inner[2], flux_inner[3], flux_inner[4],
                   normal_energy_flux, flux_inner[6], flux_inner[7], flux_inner[8],
                   flux_inner[9])
end
end # @muladd
