# Currently only used for debugging/providing info in error/warning messages
@inline get_boundary_element(boundaries::AeroUnstructuredBoundaryContainer2D, boundary_index) = boundaries.element_id[boundary_index]
