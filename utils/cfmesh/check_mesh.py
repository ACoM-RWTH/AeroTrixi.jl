#!/usr/bin/env python3
"""
Basic quality check of a 2D cfMesh airfoil mesh (ASCII OpenFOAM polyMesh),
as a substitute for OpenFOAM's checkMesh, which is not part of CF-MESH+.
Useful to inspect the cfMesh base mesh before conversion with
polymesh_to_inp.py (pass nLayers of system/meshDict, i.e., 12).

Reports
  - number of cells and their polygon types (quads; pentagons/hexagons occur
    at octree level transitions due to hanging nodes),
  - validity (consistent orientation / positive cell areas),
  - non-orthogonality of internal faces,
  - the wall-normal layer stack on the airfoil (first/last/total layer height
    and the largest height ratio between neighbouring cells, which shows the
    jump from the last layer to the core mesh),
and writes mesh_preview.png (leading edge, trailing edge, whole domain).

Usage:  python3 check_mesh.py [case_dir] [nLayers]
"""
import re
import sys
from collections import Counter

import numpy as np

case = sys.argv[1] if len(sys.argv) > 1 else "."
n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
d = case + "/constant/polyMesh/"


def read_list(name):
    s = open(d + name).read()
    s = s[s.index("}", s.index("FoamFile")) + 1:]
    s = re.sub(r"//.*", "", s)
    m = re.search(r"\n\s*(\d+)\s*\n\s*\(", s)
    return int(m.group(1)), s[m.end():]


n, s = read_list("points")
pts = np.array([t.split() for t in re.findall(r"\(([^()]+)\)", s)[:n]], float)
n, s = read_list("faces")
faces = [list(map(int, f.split())) for f in re.findall(r"\d+\(([^()]*)\)", s)[:n]]
n, s = read_list("owner")
owner = np.array(s.split()[:n], int)
n, s = read_list("neighbour")
nbr = np.array(s.split()[:n], int)
patches = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(
    r"(\w+)\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);",
    open(d + "boundary").read())}
# cfMesh writes -1 as neighbour of boundary faces
nbr = nbr[:min(st for _, st in patches.values())]

n_cells = owner.max() + 1
cell_faces = [[] for _ in range(n_cells)]
for f, c in enumerate(owner):
    cell_faces[c].append(f)
for f, c in enumerate(nbr):
    cell_faces[c].append(f)
empty = set()
for p in ("bottomEmptyFaces", "topEmptyFaces"):
    nf, st = patches[p]
    empty.update(range(st, st + nf))

print(f"cells: {n_cells}")
print("cell types (number of edges: count):",
      dict(sorted(Counter(len(cf) - 2 for cf in cell_faces).items())))
print("patches (nFaces):", {k: v[0] for k, v in patches.items()})

# ---------------------------------------------------------------- validity
nf, st = patches["bottomEmptyFaces"]
polys = [pts[faces[f]][:, :2] for f in range(st, st + nf)]


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def centroid(p):
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (3 * cr.sum())


A = np.array([signed_area(p) for p in polys])
ok = (A > 0).all() or (A < 0).all()
print(f"cell areas: consistent sign (valid) = {ok}, "
      f"min {np.abs(A).min():.3e}, max {np.abs(A).max():.3e}")

C = np.zeros((n_cells, 2))
C[owner[st:st + nf]] = [centroid(p) for p in polys]
non_orth = np.empty(len(nbr))
for f in range(len(nbr)):
    xy = np.unique(np.round(pts[faces[f]][:, :2], 14), axis=0)
    t = xy[-1] - xy[0]
    normal = np.array([t[1], -t[0]]) / np.linalg.norm(t)
    dc = C[nbr[f]] - C[owner[f]]
    non_orth[f] = np.degrees(np.arccos(min(1.0, abs(normal @ dc) / np.linalg.norm(dc))))
print(f"non-orthogonality [deg]: max {non_orth.max():.1f}, mean {non_orth.mean():.2f}, "
      f"> 70: {(non_orth > 70).sum()}")

# ----------------------------------------------------------- layer stacks
face_centre = lambda f: pts[faces[f]].mean(axis=0)[:2]


def stack(f, n_max):
    """Heights of the cells in the wall-normal column starting at wall face f."""
    c, cur, hs = owner[f], f, []
    for _ in range(n_max):
        opposite = [g for g in cell_faces[c] if g not in empty and g != cur
                    and not (set(faces[g]) & set(faces[cur]))]
        if len(cell_faces[c]) != 6 or len(opposite) != 1:
            break  # column leaves the quad (layer) region
        g = opposite[0]
        hs.append(np.linalg.norm(face_centre(g) - face_centre(cur)))
        if g >= len(nbr):
            break
        c, cur = (nbr[g] if owner[g] == c else owner[g]), g
    return np.array(hs)


nf, st = patches["airfoil"]
first, last, total, jump = [], [], [], []
for f in range(st, st + nf):
    hs = stack(f, n_layers + 3)
    if len(hs) < n_layers + 2:
        continue
    first.append(hs[0])
    last.append(hs[n_layers - 1])
    total.append(hs[:n_layers].sum())
    jump.append((hs[1:n_layers + 2] / hs[:n_layers + 1]).max())
q = lambda a: f"min {np.min(a):.2e}, median {np.median(a):.2e}, max {np.max(a):.2e}"
print(f"airfoil layer stacks ({len(first)} of {nf} wall faces, nLayers = {n_layers}):")
print("  first layer height:", q(first))
print("  last layer height :", q(last))
print("  total thickness   :", q(total))
print("  max height ratio (incl. jump to core cells):", q(jump))

# ----------------------------------------------------------------- preview
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
except ImportError:
    sys.exit(0)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
views = [((-0.02, 0.03), (-0.025, 0.025), "leading edge"),
         ((0.985, 1.01), (-0.0125, 0.0125), "trailing edge"),
         ((-55, 55), (-55, 55), "whole domain")]
for ax, (xl, yl, title) in zip(axs, views):
    ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="k",
                                     linewidths=0.3))
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")
    ax.set_title(title)
plt.tight_layout()
plt.savefig(case + "/mesh_preview.png", dpi=110)
print("wrote", case + "/mesh_preview.png")
