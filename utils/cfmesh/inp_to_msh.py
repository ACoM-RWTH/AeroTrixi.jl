#!/usr/bin/env python3
"""
Convert a 2D Abaqus mesh (.inp) as read by Trixi.jl's P4estMesh into a Gmsh
mesh file (MSH 2.2, ASCII), e.g., to inspect the mesh and its boundaries in
Gmsh (https://gmsh.info).

Input: linear quadrilaterals (CPS4, CPE4, S4, ...) and boundary node sets
(*NSET,NSET=<name>). As in Trixi.jl, an element edge belongs to a boundary if
both of its nodes are contained in the node set; additionally, only edges of a
single element (i.e., edges on the domain boundary) are considered.

Output: the quadrilaterals as physical surface "fluid" and, for each node set,
the boundary edges as 2-node line elements in a physical curve named like the
node set. Boundary edges which are not contained in any node set are written to
the physical curve "unassigned" (and reported), which helps to detect
boundaries without boundary condition.

Usage:
    python3 inp_to_msh.py <mesh.inp> [mesh.msh]
"""
import sys

inp_file = sys.argv[1]
msh_file = sys.argv[2] if len(sys.argv) > 2 else inp_file.rsplit(".", 1)[0] + ".msh"

# =============================================================================
# Read .inp
# =============================================================================
nodes = {}       # id -> (x, y, z)
quads = []       # (n1, n2, n3, n4)
node_sets = {}   # name -> set of node ids
section = None
quad_types = ("CPS4", "CPE4", "CPEG4", "S4", "M3D4", "SFM3D4")
with open(inp_file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("**"):
            continue  # comment, includes "******* E L E M E N T S *************"
        if line.startswith("*"):
            keyword = line.upper()
            if keyword.startswith("*NODE"):
                section = "node"
            elif keyword.startswith("*ELEMENT"):
                etype = keyword.split("TYPE=")[1].split(",")[0].strip()
                section = "quad" if etype.startswith(quad_types) else "skip"
            elif keyword.startswith("*NSET"):
                name = line.split("=", 1)[1].split(",")[0].strip()
                node_sets[name] = set()
                section = ("nset", name)
            else:
                section = None
            continue
        parts = [p for p in line.split(",") if p.strip()]
        if section == "node":
            nodes[int(parts[0])] = tuple(float(p) for p in parts[1:4]) + (0.0,) * (4 - len(parts))
        elif section == "quad":
            quads.append(tuple(int(p) for p in parts[1:5]))
        elif isinstance(section, tuple):
            node_sets[section[1]].update(int(p) for p in parts)

# =============================================================================
# Boundary edges
# =============================================================================
edge_count = {}
for q in quads:
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        key = (min(a, b), max(a, b))
        edge_count[key] = edge_count.get(key, 0) + 1
# keep the orientation of the edge within its element (counter-clockwise quads
# give boundary edges oriented with the domain on the left)
boundary_edges = []
for q in quads:
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        if edge_count[(min(a, b), max(a, b))] == 1:
            boundary_edges.append((a, b))

names = list(node_sets) + ["unassigned"]
edges_per_set = {name: [] for name in names}
for a, b in boundary_edges:
    owners = [name for name in node_sets if a in node_sets[name] and b in node_sets[name]]
    if len(owners) > 1:
        print(f"Warning: boundary edge ({a}, {b}) is contained in several node sets "
              f"{owners}; using {owners[0]}")
    edges_per_set[owners[0] if owners else "unassigned"].append((a, b))
if not edges_per_set["unassigned"]:
    names.remove("unassigned")

# =============================================================================
# Write MSH 2.2 (ASCII)
# =============================================================================
# physical/elementary tags: curves 1..n, surface n + 1
tag = {name: i + 1 for i, name in enumerate(names)}
surface_tag = len(names) + 1
n_elements = sum(len(e) for e in edges_per_set.values()) + len(quads)
with open(msh_file, "w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    f.write(f"$PhysicalNames\n{len(names) + 1}\n")
    for name in names:
        f.write(f'1 {tag[name]} "{name}"\n')
    f.write(f'2 {surface_tag} "fluid"\n$EndPhysicalNames\n')
    f.write(f"$Nodes\n{len(nodes)}\n")
    for i, (x, y, z) in sorted(nodes.items()):
        f.write(f"{i} {x:.15e} {y:.15e} {z:.15e}\n")
    f.write(f"$EndNodes\n$Elements\n{n_elements}\n")
    eid = 1
    for name in names:  # element type 1: 2-node line
        for a, b in edges_per_set[name]:
            f.write(f"{eid} 1 2 {tag[name]} {tag[name]} {a} {b}\n")
            eid += 1
    for q in quads:     # element type 3: 4-node quadrangle
        f.write(f"{eid} 3 2 {surface_tag} {surface_tag} {q[0]} {q[1]} {q[2]} {q[3]}\n")
        eid += 1
    f.write("$EndElements\n")

print(f"nodes: {len(nodes)}, quads: {len(quads)}, boundary edges: {len(boundary_edges)}")
for name in names:
    print(f"  physical curve '{name}': {len(edges_per_set[name])} edges")
if "unassigned" in names:
    print("Warning: some boundary edges are not contained in any node set")
print(f"wrote {msh_file}")
