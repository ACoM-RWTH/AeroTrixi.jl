# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# 2D viscous stress vector based on contracting the viscous stress tensor
# with the normalized `normal_direction` vector.
function viscous_stress_vector(u, normal_direction, equations_parabolic,
                               gradients_1, gradients_2)
    #  Normalize normal direction, should point *into* the fluid => *(-1)
    n_normal = -normal_direction / norm(normal_direction)

    tau_11, tau_12, tau_22 = viscous_stress_tensor(u, equations_parabolic,
                                                   gradients_1, gradients_2)

    # Viscous stress vector: Stress tensor * normal vector
    viscous_stress_vector_1 = tau_11 * n_normal[1] + tau_12 * n_normal[2]
    viscous_stress_vector_2 = tau_12 * n_normal[1] + tau_22 * n_normal[2]

    return (viscous_stress_vector_1, viscous_stress_vector_2)
end
end # muladd
