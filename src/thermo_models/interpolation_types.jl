# what kind of interpolation we use for the quantities
abstract type Interpolation end
struct LinearInterpolation <: Interpolation end

# do we store c_v(T) at T_min + i * ΔT, or at T_min - 0.5 ΔT + i * ΔT
# i.e. offset, or no offset w.r.t internal energy
abstract type CvTableOffset end

"""
    NoCvOffset <: CvTableOffset

Tabulate ``c_v(T)`` at the same temperatures as the internal energy,
``T_{\\min} + i \\Delta T``. See [`ThermoData1T`](@ref).
"""
struct NoCvOffset <: CvTableOffset end

"""
    CvOffset <: CvTableOffset

Tabulate ``c_v(T)`` at cell midpoints, ``T_{\\min} - \\Delta T/2 + i \\Delta T``, i.e.
offset by half a step from the internal energy grid. See [`ThermoData1T`](@ref).
"""
struct CvOffset <: CvTableOffset end

# abstract container for thermodynamic data
abstract type ThermoData end