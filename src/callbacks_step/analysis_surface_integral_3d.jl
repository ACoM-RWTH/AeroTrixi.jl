# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Compute the three components of the 2D symmetric viscous stress tensor
# (tau_11, tau_12, tau_22) based on the gradients of the velocity field.
# This is required for drag and lift coefficients based on shear stress,
# as well as for the non-integrated quantities such as
# skin friction coefficient (to be added).
function viscous_stress_tensor(u, equations_parabolic,
                               gradients_1, gradients_2, gradients_3)
    _, dv1dx, dv2dx, dv3dx, _ = convert_derivative_to_primitive(u, gradients_1,
                                                                equations_parabolic)
    _, dv1dy, dv2dy, dv3dy, _ = convert_derivative_to_primitive(u, gradients_2,
                                                                equations_parabolic)
    _, dv1dz, dv2dz, dv3dz, _ = convert_derivative_to_primitive(u, gradients_3,
                                                                equations_parabolic)

    # Components of viscous stress tensor

    # Diagonal parts
    # (4 * (v1)_x / 3 - 2 * ((v2)_y + (v3)_z)) / 3)
    tau_11 = (4 * dv1dx - 2 * (dv2dy + dv3dz)) / 3
    # (4 * (v2)_y / 3 - 2 * ((v1)_x + (v3)_z) / 3)
    tau_22 = (4 * dv2dy - 2 * (dv1dx + dv3dz)) / 3
    # (4 * (v3)_z / 3 - 2 * ((v1)_x + (v2)_y) / 3)
    tau_33 = (4 * dv3dz - 2 * (dv1dx + dv2dy)) / 3

    # Off diagonal parts, exploit that stress tensor is symmetric
    # ((v1)_y + (v2)_x)
    tau_12 = dv1dy + dv2dx # = tau_21
    # ((v1)_z + (v3)_x)
    tau_13 = dv1dz + dv3dx # = tau_31
    # ((v2)_z + (v3)_y)
    tau_23 = dv2dz + dv3dy # = tau_32

    mu = dynamic_viscosity(u, equations_parabolic)

    return mu .* (tau_11, tau_12, tau_13,
                  tau_22, tau_23,
                  tau_33)
end

# 3D viscous stress vector based on contracting the viscous stress tensor
# with the normalized `normal_direction` vector.
function viscous_stress_vector(u, normal_direction, equations_parabolic,
                               gradients_1, gradients_2, gradients_3)
    #  Normalize normal direction, should point *into* the fluid => *(-1)
    n_normal = -normal_direction / norm(normal_direction)

    tau_11, tau_12, tau_13,
    tau_22, tau_23,
    tau_33 = viscous_stress_tensor(u, equations_parabolic,
                                   gradients_1, gradients_2, gradients_3)

    # Viscous stress vector: Stress tensor * normal vector
    viscous_stress_vector_1 = tau_11 * n_normal[1] +
                              tau_12 * n_normal[2] +
                              tau_13 * n_normal[3]

    viscous_stress_vector_2 = tau_12 * n_normal[1] +
                              tau_22 * n_normal[2] +
                              tau_23 * n_normal[3]

    viscous_stress_vector_3 = tau_13 * n_normal[1] +
                              tau_23 * n_normal[2] +
                              tau_33 * n_normal[3]

    return (viscous_stress_vector_1, viscous_stress_vector_2, viscous_stress_vector_3)
end
end # muladd
