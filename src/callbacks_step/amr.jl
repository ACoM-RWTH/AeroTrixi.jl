# From Trixi.jl: "src/callbacks_step/amr.jl"
function reinitialize_boundaries!(boundary_conditions::AeroUnstructuredSortedBoundaryTypes,
                                  cache)
    # Reinitialize boundary types container because boundaries may have changed.
    return reinitialize!(boundary_conditions, cache)
end
