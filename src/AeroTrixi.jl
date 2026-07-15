"""
    AeroTrixi

High-fidelity aerodynamic simulations with Trixi.jl.
"""
module AeroTrixi

using Trixi
# using bunch of non-exported stuff from Trixi to avoid writing `Trixi.` everywhere
using Trixi: @printf, @sprintf, print_level_information,
             @trixi_timeit, @notimeit, timer,
             DiscreteCallback, SolutionAnalyzer,
             create_cache_analysis, summary_box, ncalls,
             AbstractEquations, AbstractEquationsParabolic, AbstractSemidiscretization,
             mesh_equations_solver_cache, get_tmp_cache,
             wrap_array,
             u_modified!, isfinished,
             mpi_isroot, mpi_nranks, mpi_println,
             ndofsglobal, ndofs, nelementsglobal, nelements,
             get_name, attributes,
             get_boundary_indices, get_node_coords, get_normal_direction,
             indices2direction,
             prolong2boundaries!,
             index_to_start_step_2d, index_to_start_step_3d,
             analyze_integrals, calc_error_norms,
             h5open,
             convert_derivative_to_primitive,
             viscous_stress_tensor # 2D version in main Trixi.jl

# import (not using!) functions that are extended
import Trixi: pretty_form_ascii, pretty_form_utf,
              initialize!

#viscous_stress_tensor # 3D version not in main Trixi.jl, but also currently not used

using MuladdMacro: @muladd
using StaticArrays: SVector, SMatrix, SArray, MVector, MArray
using LinearAlgebra: norm

include("auxiliary.jl")

include("callbacks_step/callbacks_step.jl")

export AnalysisSurfacePointwise, SurfacePressureCoefficient, SurfaceFrictionCoefficient,
       AnalysisCallback,
       examples_dir

end
