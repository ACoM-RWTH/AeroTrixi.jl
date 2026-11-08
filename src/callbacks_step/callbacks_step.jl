# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

include("surface_pointwise_callback.jl")
include("gridap_callback.jl")
include("relaxation_stepsize_callback.jl")
include("controller_minimum_size.jl")
end # @muladd
