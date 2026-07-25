function Trixi.digest_boundary_conditions(boundary_conditions::NamedTuple,
                                          mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                                      UnstructuredMesh2D,
                                                      T8codeMesh{2}},
                                          solver, cache)
    return AeroUnstructuredSortedBoundaryTypes(boundary_conditions, cache)
end

function Trixi.digest_boundary_conditions(boundary_conditions::NamedTuple,
                                          mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                          solver, cache)
    return AeroUnstructuredSortedBoundaryTypes(boundary_conditions, cache)
end

function digest_boundary_conditions(boundary_conditions::AeroUnstructuredSortedBoundaryTypes,
                                    mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                                UnstructuredMesh2D,
                                                T8codeMesh{2}},
                                    solver, cache)
    return boundary_conditions
end

function digest_boundary_conditions(boundary_conditions::AeroUnstructuredSortedBoundaryTypes,
                                    mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                    solver, cache)
    return boundary_conditions
end

# allow passing a single BC that get converted into a named tuple of BCs
# on (mapped) hypercube domains
function Trixi.digest_boundary_conditions(boundary_conditions,
                                          mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                                      UnstructuredMesh2D,
                                                      T8codeMesh{2}},
                                          solver, cache)
    bcs = (; x_neg = boundary_conditions, x_pos = boundary_conditions,
           y_neg = boundary_conditions, y_pos = boundary_conditions)
    return AeroUnstructuredSortedBoundaryTypes(bcs, cache)
end

function Trixi.digest_boundary_conditions(boundary_conditions,
                                          mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                          solver, cache)
    bcs = (; x_neg = boundary_conditions, x_pos = boundary_conditions,
           y_neg = boundary_conditions, y_pos = boundary_conditions,
           z_neg = boundary_conditions, z_pos = boundary_conditions)
    return AeroUnstructuredSortedBoundaryTypes(bcs, cache)
end

function print_boundary_conditions(io,
                                   semi::SemiHypMeshBCSolver{<:Any,
                                                             <:AeroUnstructuredSortedBoundaryTypes})
    @unpack boundary_conditions = semi.boundary_conditions
    summary_line(io, "boundary conditions", length(boundary_conditions))
    for (boundary_name, boundary_condition) in pairs(boundary_conditions)
        summary_line(increment_indent(io), boundary_name, typeof(boundary_condition))
    end
end
