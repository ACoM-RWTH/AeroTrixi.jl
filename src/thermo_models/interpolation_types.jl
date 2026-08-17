# what kind of interpolation we use for the quantities
abstract type Interpolation end
struct LinearInterpolation <: Interpolation end

# subtypes  whether we store c_v(T) at T_min + i * dT (NoCvOffset)
# or at T_min - 0.5 dT + i * dT (CvOffset)
# i.e. without or with an offset with respect to
# the points at which internal energy is stored
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
