# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    GridapCallback(semi; dirichlet_tags = (), interval = 1, problem_kwargs...)
    GridapCallback(semi, problem; source, write, dirichlet_tags = (), interval = 1)

Callback that solves a finite element problem with continuous Q2 elements in Gridap.jl on
the current mesh and writes the result into the DG solution (DG with `polydeg = 2`). It is
called at the start of the simulation and after every `interval` time steps; place it after
an `AMRCallback` in the `CallbackSet`, so that the solution is recomputed after mesh
changes.

The first form solves the elliptic part of the equations of `semi` (see
[`elliptic_forms`](@ref)): the problem is `GridapLinearProblem(equations; problem_kwargs...)`,
the source is [`elliptic_sources`](@ref), and the solution is written with
[`write_elliptic_solution`](@ref). The second form takes all parts explicitly:

- `problem` describes the finite element problem, e.g., a [`GridapLinearProblem`](@ref);
  see [`gridap_setup`](@ref) and [`gridap_solve`](@ref) for the interface.
- `source(u_node, x, equations)` returns an `SVector` of source fields at each DG node,
  which are transferred to the Q2 nodes with [`transfer_dg_to_q2!`](@ref) and passed to the
  problem.
- `write(u_node, q, x, equations)` returns the new state at each DG node, see
  [`transfer_q2_to_dg!`](@ref); `q2_value(q, k)` and `q2_gradient(q, k)` give the
  solution field `k` and its gradient.
- `dirichlet_tags`: boundary names with Dirichlet conditions in the Q2 space.

The node numbering, the Gridap model and spaces, and the result of
[`gridap_setup`](@ref) (e.g., a matrix factorization) are kept and only rebuilt when the
mesh changes (detected with the `revision` counter of `p4est`).

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
the stage limiter is not called). The finite element problem is then solved on each stage
value before the right-hand side is evaluated, and on the solution at the end of each step,
with the data cached by [`gridap_setup`](@ref). The callback is switched to act only at the
start and after mesh changes, where it rebuilds the cached data and updates the solution on
the new mesh; keep it after the `AMRCallback` in the `CallbackSet`.

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function gridap_stage_update end

"""
    GridapLinearProblem(bilinear_form, linear_form; factorize = lu, quadrature_degree = 4,
                        precompute_load = true)
    GridapLinearProblem(equations; factorize, quadrature_degree = 4, precompute_load = true)

Linear finite element problem `a(u, v) = l(v)` for all test functions `v`, with
homogeneous Dirichlet conditions on the `dirichlet_tags` of the [`GridapCallback`](@ref).
The forms are functions
- `bilinear_form(u, v, dΩ)`, e.g., `∫(∇(v) ⋅ ∇(u)) * dΩ`,
- `linear_form(v, sources, dΩ)`, with the source fields `sources[k]` as Gridap.jl
  functions, e.g., `∫(v * sources[1]) * dΩ`,

where `dΩ` is the Gridap.jl measure of the mesh with `quadrature_degree`. The matrix is
assembled and factorized with `factorize` (e.g., `A -> cholesky(Symmetric(A))` for a
symmetric positive definite matrix) after each mesh change.

With `precompute_load = true`, the linear form must be affine in the sources (as in the
example). Then the load vector is `b0 + Σ_k B_k f_k` with the values `f_k` of the source
field `k` at all Q2 nodes; the vector `b0` and the matrices `B_k` are assembled after each
mesh change (and checked against a direct assembly with random sources), and each solve only
needs sparse matrix-vector products and the solve with the factorization. With
`precompute_load = false`, the load vector is assembled for each solve.

The second form takes the forms of the elliptic part of `equations` from
[`elliptic_forms`](@ref); by default, the matrix is factorized with Cholesky if
[`elliptic_symmetric`](@ref) is `true`, and with LU otherwise. This form requires Gridap.jl
to be loaded (`import Gridap`).
"""
struct GridapLinearProblem{BilinearForm, LinearForm, Factorize}
    bilinear_form     :: BilinearForm
    linear_form       :: LinearForm
    factorize         :: Factorize
    quadrature_degree :: Int
    precompute_load   :: Bool
end

function GridapLinearProblem(bilinear_form, linear_form; factorize = lu,
                             quadrature_degree = 4, precompute_load = true)
    return GridapLinearProblem(bilinear_form, linear_form, factorize,
                               Int(quadrature_degree), precompute_load)
end

"""
    gridap_setup(problem, q2, model, n_sources)

Interface of the problems of a [`GridapCallback`](@ref): set up the mesh-dependent data of
`problem` (e.g., assemble and factorize a matrix) for the Q2 space `q2` created by
[`gridap_q2_space`](@ref) on the Gridap.jl `model`, with `n_sources` source fields.
Called at the start and after each mesh change; the result is passed to
[`gridap_solve`](@ref).

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function gridap_setup end

"""
    gridap_solve(problem, setup, q2, source_nodes)

Interface of the problems of a [`GridapCallback`](@ref): solve `problem` with the data
`setup` from [`gridap_setup`](@ref) for the source fields given by their values at the
nodes of the Q2 space `q2`, `source_nodes[k, node]` for the field `k` (with
[`gridap_q2_function`](@ref), they become Gridap.jl functions). Return the solution as a
Gridap.jl function, or a tuple of functions for several solution fields; functions of the
Q2 space are converted to node values directly from their degrees of freedom (see
[`gridap_node_values`](@ref)).

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function gridap_solve end

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
[`elliptic_symmetric`](@ref); a [`GridapCallback`](@ref) then solves the elliptic part with
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
  (see [`elliptic_sources`](@ref)).

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
    elliptic_symmetric(equations)

Whether the matrix of the elliptic part of `equations` is symmetric positive definite, so
that it is factorized with Cholesky instead of LU. Default: `false`.
"""
elliptic_symmetric(equations) = false
end # @muladd
