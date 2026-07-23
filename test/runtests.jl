using Test
using Aqua

using Trixi
using AeroTrixi

@testset "AeroTrixi" begin
    include(joinpath(@__DIR__, "test_spectra.jl"))
    include(joinpath(@__DIR__, "test_thermo_interpolation.jl"))
    include(joinpath(@__DIR__, "test_euler_ms1T_2D.jl"))
    include(joinpath(@__DIR__, "test_p4est.jl"))
end

@testset "Aqua" begin
    Aqua.test_all(AeroTrixi)
end
