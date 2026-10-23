# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    GridapCallback(semi, problem; source, write, dirichlet_tags = (), interval = 1)

Callback that solves a finite element problem with continuous Q2 elements in Gridap.jl on
the current mesh and writes the result into the DG solution (DG with `polydeg = 2`). It is
called at the start of the simulation and after every `interval` time steps; place it after
an `AMRCallback` in the `CallbackSet`, so that the solution is recomputed after mesh
changes.

- `problem` describes the finite element problem, e.g., a [`GridapLinearProblem`](@ref);
  see [`gridap_setup`](@ref) and [`gridap_solve`](@ref) for the interface.
- `source(u_node, x, equations)` returns an `SVector` of source fields at each DG node,
  which are transferred to the Q2 nodes with [`transfer_dg_to_q2!`](@ref) and passed to the
  problem as Gridap.jl functions.
- `write(u_node, q, x, equations)` returns the new state at each DG node, see
  [`transfer_q2_to_dg!`](@ref); `q2_value(q, k)` and `q2_gradient(q, k)` give the
  solution field `k` and its gradient. For [`CompressibleEulerPoissonEquations2D`](@ref),
  [`write_electrostatic`](@ref) sets the potential and the field.
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
    GridapLinearProblem(bilinear_form, linear_form; factorize = lu, quadrature_degree = 4)

Linear finite element problem `a(u, v) = l(v)` for all test functions `v`, with
homogeneous Dirichlet conditions on the `dirichlet_tags` of the [`GridapCallback`](@ref).
The forms are functions
- `bilinear_form(u, v, dΩ)`, e.g., `∫(∇(v) ⋅ ∇(u)) * dΩ`,
- `linear_form(v, sources, dΩ)`, with the source fields `sources[k]` as Gridap.jl
  functions, e.g., `∫(v * sources[1]) * dΩ`,

where `dΩ` is the Gridap.jl measure of the mesh with `quadrature_degree`. The matrix is
assembled and factorized with `factorize` (e.g., `A -> cholesky(Symmetric(A))` for a
symmetric positive definite matrix) after each mesh change; each solve only assembles the
right-hand side.
"""
struct GridapLinearProblem{BilinearForm, LinearForm, Factorize}
    bilinear_form     :: BilinearForm
    linear_form       :: LinearForm
    factorize         :: Factorize
    quadrature_degree :: Int
end

function GridapLinearProblem(bilinear_form, linear_form; factorize = lu,
                             quadrature_degree = 4)
    return GridapLinearProblem(bilinear_form, linear_form, factorize,
                               Int(quadrature_degree))
end

"""
    gridap_setup(problem, q2, model)

Interface of the problems of a [`GridapCallback`](@ref): set up the mesh-dependent data of
`problem` (e.g., assemble and factorize a matrix) for the Q2 space `q2` created by
[`gridap_q2_space`](@ref) on the Gridap.jl `model`. Called at the start and after each
mesh change; the result is passed to [`gridap_solve`](@ref).

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function gridap_setup end

"""
    gridap_solve(problem, setup, q2, sources)

Interface of the problems of a [`GridapCallback`](@ref): solve `problem` with the data
`setup` from [`gridap_setup`](@ref) for the source fields `sources` (a tuple of Gridap.jl
functions in the Q2 space `q2`). Return the solution as a Gridap.jl function, or a tuple
of functions for several solution fields.

This function requires Gridap.jl to be loaded (`import Gridap`).
"""
function gridap_solve end
end # @muladd
