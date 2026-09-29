"""
    AeroTrixi

High-fidelity aerodynamic simulations with Trixi.jl.
"""
module AeroTrixi

using Trixi
# using bunch of non-exported stuff from Trixi to avoid writing `Trixi.` everywhere
using Trixi: @printf, @sprintf,
             @trixi_timeit, @notimeit, timer,
             DiscreteCallback, summary_box,
             AbstractEquations, AbstractEquationsParabolic, AbstractSemidiscretization,
             AbstractCompressibleEulerMulticomponentEquations,
             ln_mean,
             mesh_equations_solver_cache, get_tmp_cache,
             wrap_array,
             derivative_discontinuity!, isfinished,
             attributes,
             get_boundary_indices, get_node_coords, get_normal_direction,
             indices2direction,
             prolong2boundaries!,
             index_to_start_step_2d, index_to_start_step_3d,
             h5open,
             convert_derivative_to_primitive,
             viscous_stress_tensor # 2D version in main Trixi.jl

# import (not using!) functions that are extended
import Trixi: pretty_form_ascii, pretty_form_utf,
              varnames, cons2prim, prim2cons, cons2entropy,
              density, pressure, temperature, density_pressure,
              energy_total, energy_kinetic, energy_internal,
              entropy, entropy_math, entropy_thermodynamic,
              ncomponents, eachcomponent,
              flux, max_abs_speed, max_abs_speeds,
              boundary_condition_slip_wall, rotate_to_x, rotate_from_x

#viscous_stress_tensor # 3D version not in main Trixi.jl, but also currently not used

using MuladdMacro: @muladd
using StaticArrays: SVector, SMatrix, SArray, MVector, MArray
using LinearAlgebra: norm
using FlowRef: ReferenceFlowQuantities, k_B

include("auxiliary.jl")

include("callbacks_step/callbacks_step.jl")
include("thermo_models/thermo_models.jl")
include("equations/equations.jl")

export AnalysisSurfacePointwise, SurfacePressureCoefficient, SurfaceFrictionCoefficient,
       SurfacePointwiseCallback,
       examples_dir

export e_rot_cont, c_rot_cont, generate_e_c_rot_cont
export e_vibr_iho, c_vibr_iho, generate_e_c_vibr_iho
export generate_e_vibr_arr_harmonic_cutoff_K, generate_e_vibr_arr_anharmonic_cutoff_K
export e_vibr_from_array, c_vibr_from_array, generate_e_c_vibr_from_array
export LinearInterpolation, CvOffset, NoCvOffset
export CompressibleEulerEquationsMs1T2D
export flux_oblapenko_etal, flux_oblapenko_etal_taylor

end
