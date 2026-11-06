# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    GridapCallback(semi; dirichlet_tags = (), interval = 1, quadrature_degree = 4)

Callback that solves the elliptic part of the equations of `semi` (see
[`elliptic_forms`](@ref)) with continuous Q2 elements in Gridap.jl on the current mesh and
writes the result into the DG solution (DG with `polydeg = 2`). It is called at the start
of the simulation and after every `interval` time steps; place it after an `AMRCallback` in
the `CallbackSet`, so that the solution is recomputed after mesh changes.

- The sources [`elliptic_sources`](@ref) are evaluated at the DG nodes and transferred to
  the Q2 nodes with [`transfer_dg_to_q2!`](@ref).
- The weak forms of [`elliptic_forms`](@ref) are integrated with `quadrature_degree`, with
  homogeneous Dirichlet conditions on the boundaries named in `dirichlet_tags`. The linear
  form must be affine in the sources.
- The solution is written into the DG solution with [`write_elliptic_solution`](@ref) and
  [`transfer_q2_to_dg!`](@ref).

After each mesh change (detected with the `revision` counter of `p4est`), the callback
rebuilds the node numbering, the Gridap.jl model and space, the matrix with its
factorization (Cholesky if [`elliptic_spd`](@ref), LU otherwise), and the matrices
that map the source values to the load vector. A solve then only needs sparse
matrix-vector products and the solve with the factorization.

With the callback alone, the solution is updated once per time step and constant during the
Runge-Kutta stages, which limits the coupled time integration to first order. For the full
order of the method, update it on each stage with [`gridap_stage_update`](@ref); the
callback then only acts at the start and after mesh changes.

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function GridapCallback end

"""
    gridap_stage_update(gridap_callback)

Return the update of a [`GridapCallback`](@ref) for use as `stage_limiter!` of the time
integration method, e.g.,
```julia
CarpenterKennedy2N54(; stage_limiter! = gridap_stage_update(gridap_callback),
                     williamson_condition = false)
```
(SSPRK and low-storage methods of OrdinaryDiffEq.jl; with `williamson_condition = true`,
the stage limiter is not called). The elliptic problem is then solved on each stage value
before the right-hand side is evaluated, and on the solution at the end of each step, with
the cached matrix factorization. The callback is switched to act only at the start and after
mesh changes, where it rebuilds the cached data and updates the solution on the new mesh;
keep it after the `AMRCallback` in the `CallbackSet`.

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function gridap_stage_update end

###############################################################################
# Elliptic part of equations

"""
    Q2Field(; components = 1)

Field of the elliptic part of equations (see [`elliptic_fields`](@ref)) in the continuous Q2
space on the nodes of the DG solution. Currently, only scalar fields (`components = 1`) are
supported.
"""
struct Q2Field
    components::Int
end
Q2Field(; components = 1) = Q2Field(components)

"""
    elliptic_fields(equations)

Unknowns of the elliptic part of `equations` as a named tuple of their spaces, e.g.,
`(phi = Q2Field(),)` for a potential. A system such as the Stokes problem would have several
fields, e.g., a velocity and a pressure.

Equations with an elliptic part implement `elliptic_fields`, [`elliptic_sources`](@ref),
[`elliptic_forms`](@ref), and [`write_elliptic_solution`](@ref), and optionally
[`elliptic_spd`](@ref); a [`GridapCallback`](@ref) then solves the elliptic part with
`GridapCallback(semi; dirichlet_tags)`.
"""
function elliptic_fields end

"""
    elliptic_sources(u, x, equations)

Source fields of the elliptic part of `equations` at a DG node with the state `u` and the
coordinates `x`, as an `SVector`; they are transferred to the Q2 nodes and passed to the
linear form of [`elliptic_forms`](@ref).
"""
function elliptic_sources end

"""
    elliptic_forms(equations, gridap)

Weak forms `(a, l)` of the elliptic part of `equations`, written with the operators of the
Gridap.jl module `gridap` (e.g., `∫, ∇, ⋅ = gridap.∫, gridap.∇, gridap.:⋅`), so that the
equations do not depend on Gridap.jl:
- `a(us, vs, dΩ)`: bilinear form for the tuples of unknowns `us` and test functions `vs`
  (one entry per field of [`elliptic_fields`](@ref)),
- `l(vs, sources, dΩ)`: linear form with the source fields `sources[k]`
  (see [`elliptic_sources`](@ref)); it must be affine in the sources.

`dΩ` is the measure of the mesh. Dirichlet conditions are homogeneous, on the boundaries
given to the [`GridapCallback`](@ref).
"""
function elliptic_forms end

"""
    write_elliptic_solution(u, q, x, equations)

New state at a DG node with the state `u` and the coordinates `x` from the solution of the
elliptic part of `equations`; `q2_value(q, k)` and `q2_gradient(q, k)` give the field `k` of
[`elliptic_fields`](@ref) and its gradient at the node.
"""
function write_elliptic_solution end

"""
    elliptic_spd(equations)

Whether the matrix of the elliptic part of `equations` is symmetric positive definite, so
that it is factorized with Cholesky instead of LU. Default: `false`.
"""
elliptic_spd(equations) = false
end # @muladd
