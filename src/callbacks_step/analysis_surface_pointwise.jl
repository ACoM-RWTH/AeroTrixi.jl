# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains callbacks that are performed on the surface like computation of
# pointwise surface forces.

"""
    SurfacePointwise{Variable, NBoundaries}(boundary_symbol_or_boundary_symbols,
                                                    variable, output_directory = "out")

This struct is used to compute pointwise surface values of a quantity of interest `variable` alongside
the boundary/boundaries associated with particular names given in `boundary_symbols`.
For instance, this can be used to compute the surface pressure coefficient [`SurfacePressureCoefficient`](@ref) or
surface friction coefficient [`SurfaceFrictionCoefficient`](@ref) of e.g. an 2D airfoil with the boundary
names `:AirfoilTop`, `:AirfoilBottom` which would be supplied as 
`boundary_symbols = (:AirfoilTop, :AirfoilBottom)`.
A single boundary name can also be supplied, e.g. `boundary_symbols = (:AirfoilTop,)`.

- `boundary_symbols::NTuple{NBoundaries, Symbol}`: Name(s) of the boundary/boundaries
  where the quantity of interest is computed
- `variable::Variable`: Quantity of interest, like lift or drag
- `output_directory = "out"`: Directory where the pointwise value files are stored.
"""
struct SurfacePointwise{Variable, NBoundaries}
    variable::Variable # Quantity of interest, like lift or drag
    boundary_symbols::NTuple{NBoundaries, Symbol} # Name(s) of the boundary/boundaries
    output_directory::String

    function SurfacePointwise(boundary_symbols::NTuple{NBoundaries, Symbol},
                              variable,
                              output_directory = "out") where {NBoundaries}
        return new{typeof(variable), NBoundaries}(variable, boundary_symbols,
                                                  output_directory)
    end
end

struct FlowState{RealT <: Real}
    rho_inf::RealT
    u_inf::RealT
end

struct SurfacePressureCoefficient{RealT <: Real}
    p_inf::RealT # Free stream pressure
    flow_state::FlowState{RealT}
end

struct SurfaceFrictionCoefficient{RealT <: Real} <: VariableViscous
    flow_state::FlowState{RealT}
end

"""
    SurfacePressureCoefficient(p_inf, rho_inf, u_inf)

Compute the surface pressure coefficient
```math
C_p \\coloneqq \\frac{p - p_{\\infty}}
                     {0.5 \\rho_{\\infty} U_{\\infty}^2}
```
based on the pressure distribution along a boundary.
Supposed to be used in conjunction with [`SurfacePointwise`](@ref)
which stores the boundary information and semidiscretization.

- `p_inf::Real`: Free-stream pressure
- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
"""
function SurfacePressureCoefficient(p_inf, rho_inf, u_inf)
    return SurfacePressureCoefficient(p_inf, FlowState(rho_inf, u_inf))
end

"""
SurfaceFrictionCoefficient(rho_inf, u_inf)

Compute the surface skin friction coefficient
```math
C_f \\coloneqq \\frac{\\boldsymbol \\tau_w  \\boldsymbol n^\\perp}
                     {0.5 \\rho_{\\infty} U_{\\infty}^2 L_{\\infty}}
```
based on the wall shear stress vector ``\\tau_w`` along a boundary.
Supposed to be used in conjunction with [`SurfacePointwise`](@ref)
which stores the boundary information and semidiscretization.

- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
"""
function SurfaceFrictionCoefficient(rho_inf, u_inf)
    return SurfaceFrictionCoefficient(FlowState(rho_inf, u_inf))
end

# Compute local pressure coefficient.
# Works for both purely hyperbolic and hyperbolic-parabolic systems.
# C_p(x) = (p(x) - p_inf) / (0.5 * rho_inf * u_inf^2)
function (pressure_coefficient::SurfacePressureCoefficient)(u, equations)
    p = pressure(u, equations)
    @unpack p_inf = pressure_coefficient
    @unpack rho_inf, u_inf = pressure_coefficient.flow_state
    return (p - p_inf) / (0.5 * rho_inf * u_inf^2)
end

varname(::Any) = @assert false "Surface variable name not assigned" # This makes sure default behaviour is not overwriting
varname(pressure_coefficient::SurfacePressureCoefficient) = "CP_x"
varname(friction_coefficient::SurfaceFrictionCoefficient) = "CF_x"

# Helper function that saves a space-dependent quantity `values`
# at every solution/quadrature point `coordinates` at 
# time `t` and iteration `iter` to disk.
# The file is written to the `output_directory` with name `varname` in HDF5 (.h5) format.
# The latter two are retrieved from the `surface_variable`,
# an instantiation of `SurfacePointwise`.
function save_pointwise_file(output_directory, varname,
                             element_indices, node_counter, coordinates, values,
                             t, iter)
    n_points = length(values)

    filename = joinpath(output_directory, varname) * @sprintf("_%06d.h5", iter)

    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["n_points"] = n_points
        attributes(file)["variable_name"] = varname

        file["time"] = t
        file["timestep"] = iter
        file["element_indices"] = element_indices
        file["node_counter"] = node_counter
        file["point_coordinates"] = coordinates
        file["point_data"] = values
    end
end

function pretty_form_ascii(::SurfacePointwise{<:SurfacePressureCoefficient{<:Any}})
    "CP(x)"
end
function pretty_form_utf(::SurfacePointwise{<:SurfacePressureCoefficient{<:Any}})
    "CP(x)"
end

function pretty_form_ascii(::SurfacePointwise{<:SurfaceFrictionCoefficient{<:Any}})
    "CF(x)"
end
function pretty_form_utf(::SurfacePointwise{<:SurfaceFrictionCoefficient{<:Any}})
    "CF(x)"
end

include("analysis_surface_pointwise_2d.jl")
include("analysis_surface_pointwise_3d.jl")
end # muladd
