# what kind of interpolation we use for the quantities
abstract type Interpolation end
struct LinearInterpolation <: Interpolation end

# do we store c_v(T) at T_min + i * ΔT, or at T_min - 0.5 ΔT + i * ΔT
# i.e. offset, or no offset w.r.t internal energy
abstract type CvTableOffset end
struct NoCvOffset <: CvTableOffset end
struct CvOffset <: CvTableOffset end

# abstract container for thermodynamic data
abstract type ThermoData end