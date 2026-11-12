# Trixi.jl-Gridap.jl bridge

The functions and types of the bridge between the DG solver of Trixi.jl and continuous
finite elements of Gridap.jl. The [guide](../trixi_gridap_bridge.md) explains the
foundations and the usage. The Gridap.jl parts are available when Gridap.jl is loaded
(`import Gridap`).

## Meshes

```@autodocs
Modules = [AeroTrixi]
Pages = ["meshes/p4est_mesh_from_vertices.jl"]
```

## Continuous Q2 nodes and Gridap.jl spaces

```@autodocs
Modules = [AeroTrixi]
Pages = ["meshes/lnodes_q2.jl"]
```

## Transfers between DG and Q2

```@autodocs
Modules = [AeroTrixi]
Pages = ["meshes/q2_transfer.jl"]
```

## Callback and elliptic part of equations

```@autodocs
Modules = [AeroTrixi]
Pages = ["callbacks_step/gridap_callback.jl"]
```

## Euler-Poisson equations

```@autodocs
Modules = [AeroTrixi]
Pages = ["equations/compressible_euler_poisson_2d.jl"]
```
