# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Compute local friction coefficient.
# Works only in conjunction with a hyperbolic-parabolic system.
# C_f(x) = (tau_w(x) * n_perp(x)) / (0.5 * rho_inf * u_inf^2)
function (surface_friction::SurfaceFrictionCoefficient)(u, normal_direction, x, t,
                                                        equations_parabolic,
                                                        gradients_1, gradients_2)
    viscous_stress_vector_ = viscous_stress_vector(u, normal_direction,
                                                   equations_parabolic,
                                                   gradients_1, gradients_2)
    @unpack rho_inf, u_inf = surface_friction.flow_state

    # Normalize as `normal_direction` is not necessarily a unit vector
    n = normal_direction / norm(normal_direction)
    # Tangent vector = perpendicular vector to normal vector
    t = (-n[2], n[1])
    return (viscous_stress_vector_[1] * t[1] +
            viscous_stress_vector_[2] * t[2]) /
           (0.5 * rho_inf * u_inf^2)
end

# Compute and save to disk a space-dependent `surface_variable`.
# For the purely hyperbolic, i.e., non-parabolic case, this is for instance 
# the pressure coefficient `SurfacePressureCoefficient`.
# The boundary/boundaries along which this quantity is to be integrated is determined by
# `boundary_symbols`, which is retrieved from `surface_variable`.
function analyze(surface_variable::AnalysisSurfacePointwise, du, u, t,
                 mesh::P4estMesh{2},
                 equations, dg::DGSEM, cache, semi, iter)
    @unpack boundaries = cache
    @unpack node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    boundary_indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    # Restore boundary values for parabolic equations
    # which overwrite the solution boundary values with the gradients
    if semi isa SemidiscretizationHyperbolicParabolic
        prolong2boundaries!(nothing, cache, u, mesh, equations, dg)
    end

    dim = 2 # Follows from mesh dispatch 
    n_nodes = nnodes(dg)
    n_boundary_elements = length(boundary_indices)

    # Store element indices of nodes for convenient postprocessing
    # In 2D, the boundaries are lines => multiply with number of nodes
    element_indices = Vector{Int}(undef, n_boundary_elements * n_nodes^2)
    # Store unqiue node counter to distinguish nodes at the same spatial position
    node_counter = Vector{Int}(undef, n_boundary_elements * n_nodes^2)
    # Physical coordinates of boundary indices
    coordinates = Matrix{real(dg)}(undef, n_boundary_elements * n_nodes, dim)
    # Variable values at boundary indices
    values = Vector{real(dg)}(undef, n_boundary_elements * n_nodes)

    index_range = eachnode(dg)
    global_node_counter = 1 # Keeps track of solution point number on the surface
    for boundary in boundary_indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in index_range
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg,
                                         node_index, boundary)

            x = get_node_coords(node_coordinates, equations, dg,
                                i_node, j_node, element)
            value = variable(u_node, equations)

            element_indices[global_node_counter] = element
            node_counter[global_node_counter] = global_node_counter
            coordinates[global_node_counter, 1] = x[1]
            coordinates[global_node_counter, 2] = x[2]
            values[global_node_counter] = value

            i_node += i_node_step
            j_node += j_node_step
            global_node_counter += 1
        end
    end

    # Save to disk
    save_pointwise_file(surface_variable.output_directory, varname(variable),
                        element_indices, node_counter, coordinates, values,
                        t, iter)
end

# Compute and save to disk a space-dependent `surface_variable`.
# For the purely hyperbolic-parabolic case, this is may be for instance 
# the surface skin fricition coefficient `SurfaceFrictionCoefficient`.
# The boundary/boundaries along which this quantity is to be integrated is determined by
# `boundary_symbols`, which is retrieved from `surface_variable`.
function analyze(surface_variable::AnalysisSurfacePointwise{Variable},
                 du, u, t, mesh::P4estMesh{2},
                 equations, equations_parabolic,
                 dg::DGSEM, cache, semi,
                 cache_parabolic, iter) where {Variable <: VariableViscous}
    @unpack boundaries = cache
    @unpack node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    boundary_indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    # Restore boundary values for parabolic equations
    # which overwrite the solution boundary values with the gradients
    prolong2boundaries!(nothing, cache, u, mesh, equations, dg)

    dim = 2 # Follows from mesh dispatch 
    n_nodes = nnodes(dg)
    n_boundary_elements = length(boundary_indices)

    # Store element indices of nodes for convenient postprocessing
    # In 2D, the boundaries are lines => multiply with number of nodes
    element_indices = Vector{Int}(undef, n_boundary_elements * n_nodes^2)
    # Store unqiue node counter to distinguish nodes at the same spatial position
    node_counter = Vector{Int}(undef, n_boundary_elements * n_nodes^2)
    # Physical coordinates of boundary indices
    coordinates = Matrix{real(dg)}(undef, n_boundary_elements * n_nodes, dim)
    # Variable values at boundary indices
    values = Vector{real(dg)}(undef, n_boundary_elements * n_nodes)

    # Additions for parabolic
    @unpack parabolic_container = cache_parabolic
    @unpack gradients = parabolic_container

    gradients_x, gradients_y = gradients

    index_range = eachnode(dg)
    global_node_counter = 1 # Keeps track of solution point number on the surface
    for boundary in boundary_indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in index_range
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg,
                                         node_index, boundary)

            x = get_node_coords(node_coordinates, equations, dg,
                                i_node, j_node, element)

            gradients_1 = get_node_vars(gradients_x, equations_parabolic, dg,
                                        i_node, j_node, element)
            gradients_2 = get_node_vars(gradients_y, equations_parabolic, dg,
                                        i_node, j_node, element)

            # Extract normal direction at nodes which points from the 
            # fluid cells *outwards*, i.e., *into* the structure.
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node, element)

            # Integral over whole boundary surface
            value = variable(u_node, normal_direction, x, t, equations_parabolic,
                             gradients_1, gradients_2)

            element_indices[global_node_counter] = element
            node_counter[global_node_counter] = global_node_counter
            coordinates[global_node_counter, 1] = x[1]
            coordinates[global_node_counter, 2] = x[2]
            values[global_node_counter] = value

            i_node += i_node_step
            j_node += j_node_step
            global_node_counter += 1
        end
    end

    # Save to disk
    save_pointwise_file(surface_variable.output_directory, varname(variable),
                        element_indices, node_counter, coordinates, values,
                        t, iter)
end
end # muladd
