#!/usr/bin/env python3
"""
Convert the 2D cfMesh airfoil mesh (ASCII OpenFOAM polyMesh written by
cartesian2DMesh) into an all-quad, conforming Abaqus .inp mesh for Trixi.jl's
P4estMesh, with boundaries given as node sets (*NSET,NSET=airfoil / farfield).

Why splitting is needed
-----------------------
cfMesh meshes are octree based. At refinement-level transitions the coarse cell
keeps a hanging node on its edge, so OpenFOAM stores it as a pentagon/hexagon.
p4est needs conforming quads. Splitting every n-gon into n quads (edge nodes +
one centre node) is conforming (each edge is split exactly once) and turns the
whole mesh into quads. This halves all cell sizes, so the cfMesh base mesh is
generated one level coarser (see system/meshDict).

Splitting details
-----------------
- Wall-normal edges of the airfoil boundary-layer stack are split at the
  geometric point 1/(1 + sqrt(r)) (measured from the wall side, r = layer
  growth ratio), so n layers with ratio r become 2n layers with ratio sqrt(r).
- New nodes on boundary edges are placed on the exact geometry (airfoil spline
  of generate_geometry.py, far-field circle) instead of the straight edge
  midpoint. On the airfoil this displacement is blended out linearly over the
  layer stack (mid-edge nodes of all layer interfaces are shifted with weight
  1 - s/S, s = wall distance, S = layer thickness), so the thin near-wall
  cells are shifted rigidly instead of being squeezed or inverted.
- Centre nodes of quads are placed by transfinite (Coons) interpolation of the
  four edge nodes; centre nodes of n-gons at the vertex average.

Usage:
    python3 polymesh_to_inp.py <case_dir> <output.inp> [nLayers] [thicknessRatio]
e.g.
    python3 polymesh_to_inp.py out_oat15a oat15a_airfoil.inp 10 1.44

In Trixi.jl:
    mesh = P4estMesh{2}("oat15a_airfoil.inp", boundary_symbols = [:airfoil, :farfield])
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_geometry as geo  # noqa: E402

case = sys.argv[1]
out_file = sys.argv[2]
n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 1.44
coord_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "oat15a_coordinates_c015.dat")

# =============================================================================
# Read polyMesh
# =============================================================================
d = os.path.join(case, "constant", "polyMesh")


def read_list(name):
    s = open(os.path.join(d, name)).read()
    s = s[s.index("}", s.index("FoamFile")) + 1:]
    s = re.sub(r"//.*", "", s)
    m = re.search(r"\n\s*(\d+)\s*\n\s*\(", s)
    return int(m.group(1)), s[m.end():]


n, s = read_list("points")
pts3 = np.array([t.split() for t in re.findall(r"\(([^()]+)\)", s)[:n]], float)
n, s = read_list("faces")
faces = [list(map(int, f.split())) for f in re.findall(r"\d+\(([^()]*)\)", s)[:n]]
n, s = read_list("owner")
owner = np.array(s.split()[:n], int)
patches = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(
    r"(\w+)\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);",
    open(os.path.join(d, "boundary")).read())}
for p in ("airfoil", "farfield", "bottomEmptyFaces"):
    assert p in patches, f"patch {p} missing in polyMesh/boundary"

# 2D mesh = bottom plane (z = z_min)
z_min = pts3[:, 2].min()
is_bottom = np.abs(pts3[:, 2] - z_min) < 1e-9 * max(1.0, np.ptp(pts3[:, 2]))
pts = pts3[:, :2].copy()


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


nf, st = patches["bottomEmptyFaces"]
cells = []
for f in range(st, st + nf):
    v = faces[f]
    if signed_area(pts[v]) < 0.0:
        v = v[::-1]  # counter-clockwise, as required by Abaqus/p4est
    cells.append(v)
n_cells = len(cells)

# edges
edge_key = lambda a, b: (a, b) if a < b else (b, a)
edge_cells = {}
for c, v in enumerate(cells):
    for i in range(len(v)):
        edge_cells.setdefault(edge_key(v[i], v[(i + 1) % len(v)]), []).append(c)

boundary_edges = {}
for name in ("airfoil", "farfield"):
    nf_, st_ = patches[name]
    for f in range(st_, st_ + nf_):
        a, b = [i for i in faces[f] if is_bottom[i]]
        k = edge_key(a, b)
        assert len(edge_cells[k]) == 1, "boundary edge with two cells"
        boundary_edges[k] = name
n_bnd_open = sum(1 for k, cs in edge_cells.items() if len(cs) == 1) - len(boundary_edges)
assert n_bnd_open == 0, f"{n_bnd_open} boundary edges belong to no named patch"

# =============================================================================
# Boundary-layer columns on the airfoil
# =============================================================================
w_split = 1.0 / (1.0 + np.sqrt(ratio))  # split fraction of wall-normal edges
edge_split = {}     # edge key -> (start node, fraction from start node)
interface_edges = []  # per column: list of (edge key, wall distance s_k)
column_of_wall_edge = {}
n_conflicts = 0
for k_wall, name in boundary_edges.items():
    if name != "airfoil":
        continue
    c = edge_cells[k_wall][0]
    inner = k_wall
    s_wall = 0.0
    column = []
    for layer in range(n_layers):
        v = cells[c]
        if len(v) != 4:
            break
        i = [j for j in range(4) if edge_key(v[j], v[(j + 1) % 4]) == inner][0]
        a, b = v[i], v[(i + 1) % 4]          # inner edge (a -> b, ccw)
        qb, qa = v[(i + 2) % 4], v[(i + 3) % 4]  # outer edge: a-qa and b-qb are sides
        for p0, p1 in ((a, qa), (b, qb)):
            k = edge_key(p0, p1)
            if k in edge_split and edge_split[k][0] != p0:
                n_conflicts += 1
            else:
                edge_split[k] = (p0, w_split)
        outer = edge_key(qa, qb)
        s_wall += np.linalg.norm(0.5 * (pts[qa] + pts[qb]) - 0.5 * (pts[a] + pts[b]))
        column.append((outer, s_wall))
        nb = [cc for cc in edge_cells[outer] if cc != c]
        if not nb:
            break
        c, inner = nb[0], outer
    column_of_wall_edge[k_wall] = column
print(f"layer columns: {len(column_of_wall_edge)}, "
      f"min/max depth {min(map(len, column_of_wall_edge.values()))}/"
      f"{max(map(len, column_of_wall_edge.values()))}, "
      f"orientation conflicts: {n_conflicts}")
# The first layer height is a result of cfMesh's layer generation and the
# geometric splitting below; it cannot be prescribed.
first_layer = np.array([w_split * col[0][1] for col in column_of_wall_edge.values() if col])
total_layer = np.array([col[-1][1] for col in column_of_wall_edge.values() if col])
print(f"after splitting: {2 * n_layers} layers with ratio {np.sqrt(ratio):.4g}, "
      f"first layer height min {first_layer.min():.2e}, "
      f"median {np.median(first_layer):.2e}, max {first_layer.max():.2e}; "
      f"total layer thickness median {np.median(total_layer):.2e}")

# =============================================================================
# Exact geometry for boundary nodes
# =============================================================================
airfoil_dense = geo.airfoil_polyline(coord_file, n_airfoil=40000, n_trailing_edge=200)
seg_a = airfoil_dense
seg_b = np.roll(airfoil_dense, -1, axis=0)


def project_airfoil(p):
    ab = seg_b - seg_a
    t = np.clip(np.einsum("ij,ij->i", p - seg_a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
    q = seg_a + t[:, None] * ab
    return q[np.argmin(np.einsum("ij,ij->i", q - p, q - p))]


def project_farfield(p):
    c = np.array([geo.x_offset, geo.y_offset])
    r = p - c
    return c + geo.radius * r / np.linalg.norm(r)


# =============================================================================
# New nodes: edge nodes and cell centres
# =============================================================================
nodes = [p for p in pts]          # keep original node numbering (0-based)
edge_node = {}
for k in edge_cells:
    if k in edge_split:
        p0, w = edge_split[k]
        p1 = k[0] if k[1] == p0 else k[1]
        x = pts[p0] + w * (pts[p1] - pts[p0])
    else:
        x = 0.5 * (pts[k[0]] + pts[k[1]])
    edge_node[k] = len(nodes)
    nodes.append(x)
nodes = np.array(nodes)

max_shift = {"airfoil": 0.0, "farfield": 0.0}
for k, name in boundary_edges.items():
    i = edge_node[k]
    x = nodes[i]
    xp = project_airfoil(x) if name == "airfoil" else project_farfield(x)
    delta = xp - x
    max_shift[name] = max(max_shift[name], np.linalg.norm(delta))
    nodes[i] = xp
    if name == "airfoil":  # blend the displacement out over the layer stack
        column = column_of_wall_edge[k]
        if column:
            s_total = column[-1][1]
            for k_int, s_k in column[:-1]:
                nodes[edge_node[k_int]] += delta * (1.0 - s_k / s_total)
print(f"max boundary node shift onto exact geometry: airfoil {max_shift['airfoil']:.2e}, "
      f"farfield {max_shift['farfield']:.2e}")


def edge_fraction(i, j, e):
    """Position of edge node e along edge i -> j as a fraction from i."""
    a, b = nodes[i], nodes[j]
    return float(np.dot(nodes[e] - a, b - a) / np.dot(b - a, b - a))


nodes = list(nodes)
quads = []
for v in cells:
    m = len(v)
    e = [edge_node[edge_key(v[i], v[(i + 1) % m])] for i in range(m)]
    if m == 4:
        # Coons patch: xi along v0 -> v1, eta along v0 -> v3
        P = [np.asarray(nodes[i]) for i in v]
        E = [np.asarray(nodes[i]) for i in e]  # e0: v0v1, e1: v1v2, e2: v2v3, e3: v3v0
        xi = 0.5 * (edge_fraction(v[0], v[1], e[0]) + edge_fraction(v[3], v[2], e[2]))
        eta = 0.5 * (edge_fraction(v[0], v[3], e[3]) + edge_fraction(v[1], v[2], e[1]))
        bilinear = ((1 - xi) * (1 - eta) * P[0] + xi * (1 - eta) * P[1]
                    + xi * eta * P[2] + (1 - xi) * eta * P[3])
        centre = ((1 - eta) * E[0] + eta * E[2] + (1 - xi) * E[3] + xi * E[1]
                  - bilinear)
    else:
        centre = np.mean([nodes[i] for i in v], axis=0)
    c = len(nodes)
    nodes.append(centre)
    for i in range(m):
        quads.append((v[i], e[i], c, e[i - 1]))  # counter-clockwise
nodes = np.array(nodes)

# drop nodes that are not used (top-plane points of the polyMesh)
used = np.zeros(len(nodes), bool)
used[np.array(quads).ravel()] = True
new_id = -np.ones(len(nodes), int)
new_id[used] = np.arange(1, used.sum() + 1)  # 1-based for Abaqus
quads = new_id[np.array(quads)]
nodes = nodes[used]

# node sets
node_sets = {}
for k, name in boundary_edges.items():
    node_sets.setdefault(name, set()).update(
        new_id[[k[0], k[1], edge_node[k]]].tolist())

# =============================================================================
# Quality of the quads
# =============================================================================
Q = nodes[quads - 1]
cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
sj = []
for i in range(4):
    e1 = Q[:, (i + 1) % 4] - Q[:, i]
    e2 = Q[:, i - 1] - Q[:, i]
    sj.append(cross(e1, e2) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)))
sj = np.min(sj, axis=0)
print(f"quads: {len(quads)}, nodes: {len(nodes)}")
print(f"scaled Jacobian (min over corners): min {sj.min():.3f}, "
      f"< 0.5: {(sj < 0.5).sum()}, <= 0: {(sj <= 0).sum()}")
assert (sj > 0).all(), "inverted or degenerate quads"

# boundary edges per set in the final mesh (as Trixi assigns them)
bnd_count = {}
for name, ns in node_sets.items():
    cnt = 0
    for q in quads:
        for i in range(4):
            if q[i] in ns and q[(i + 1) % 4] in ns:
                cnt += 1
    bnd_count[name] = cnt
print("element edges with both nodes in node set:", bnd_count,
      "(expected:", {k: 2 * v for k, v in
                     {n_: sum(1 for x in boundary_edges.values() if x == n_)
                      for n_ in ("airfoil", "farfield")}.items()}, ")")

# =============================================================================
# Write Abaqus .inp
# =============================================================================
with open(out_file, "w") as f:
    f.write("*Heading\n")
    f.write(" ONERA OAT 15A, cfMesh (cartesian2DMesh) mesh converted by polymesh_to_inp.py\n")
    f.write("*NODE\n")
    for i, (x, y) in enumerate(nodes, start=1):
        f.write(f"{i}, {x:.15e}, {y:.15e}, 0.0\n")
    f.write("******* E L E M E N T S *************\n")
    f.write("*ELEMENT, type=CPS4, ELSET=Surface1\n")
    for i, q in enumerate(quads, start=1):
        f.write(f"{i}, {q[0]}, {q[1]}, {q[2]}, {q[3]}\n")
    for name in ("airfoil", "farfield"):
        f.write(f"*NSET,NSET={name}\n")
        ids = sorted(node_sets[name])
        for j in range(0, len(ids), 10):
            f.write(", ".join(map(str, ids[j:j + 10])) + ",\n")
print(f"wrote {out_file}")
