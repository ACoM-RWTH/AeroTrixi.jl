# Currently only used for debugging/providing info in error/warning messages
@inline get_boundary_element(boundaries::P4estBoundaryContainer, boundary_index) = boundaries.neighbor_ids[boundary_index]
