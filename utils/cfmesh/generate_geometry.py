#!/usr/bin/env python3
"""
Generate the cfMesh input geometry for the ONERA OAT 15A airfoil in a circular
far field.

cfMesh's 2D mesher (cartesian2DMesh) expects a "ribbon": the 2D boundary curves
in the x-y plane, extruded in the z direction. This script writes such a ribbon
as a multi-solid ASCII STL; each solid becomes one patch:

    airfoil   - airfoil surface incl. the blunt trailing edge
    farfield  - circle with radius 50 chords

Geometry conventions:
    - leading edge at (0, 0), trailing edge at x = 1 (unit chord)
    - coordinates are given for chord 0.15 and rescaled by 1/0.15
    - far-field circle centred at the leading edge

Usage:
    python3 generate_geometry.py [coordinates.dat] [output.stl]
"""
import os
import sys

import numpy as np
from scipy.interpolate import splev, splprep

# ============================================================================
# PARAMETERS
# ============================================================================
original_chord = 0.15            # chord of the tabulated coordinates
scale = 1.0 / original_chord
chord = 1.0
radius = 50.0 * chord            # far-field radius
x_offset, y_offset = 0.0, 0.0    # circle centre (= leading edge)

n_airfoil = 1200                 # number of points on airfoil spline (STL resolution)
n_trailing_edge = 6              # segments on the blunt trailing-edge base
n_farfield = 720                 # segments on the far-field circle

z_min, z_max = -0.5, 0.5         # extrusion extent of the ribbon (z direction)

# ============================================================================
# AIRFOIL CURVE
# ============================================================================
def airfoil_polyline(coord_file, n_airfoil=n_airfoil, n_trailing_edge=n_trailing_edge):
    """Closed airfoil polygon (unit chord), clockwise:
    lower TE -> LE -> upper TE -> blunt TE base (without repeating the start).

    One interpolating cubic spline is laid through all 300 tabulated points
    (lower surface TE -> LE, upper surface LE -> TE).
    The spline is re-sampled with a cosine distribution on each half, i.e.,
    clustered towards LE and TE.
    """
    xy = np.loadtxt(coord_file) * scale
    assert xy.shape == (300, 2)
    tck, _ = splprep([xy[:, 0], xy[:, 1]], s=0.0, k=3)

    # accurate arc-length parametrisation
    u_fine = np.linspace(0.0, 1.0, 200001)
    x_f, y_f = splev(u_fine, tck)
    ds = np.hypot(np.diff(x_f), np.diff(y_f))
    s_fine = np.concatenate(([0.0], np.cumsum(ds)))
    s_fine /= s_fine[-1]

    s_le = s_fine[np.argmin(x_f)]
    n_half = n_airfoil // 2
    t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_half + 1)))
    s_target = np.concatenate((s_le * t, s_le + (1.0 - s_le) * t[1:]))
    u_target = np.interp(s_target, s_fine, u_fine)
    x_a, y_a = splev(u_target, tck)
    airfoil = np.column_stack((x_a, y_a))

    # close the blunt trailing edge: upper TE -> lower TE
    te_upper, te_lower = airfoil[-1], airfoil[0]
    w = np.linspace(0.0, 1.0, n_trailing_edge + 1)[1:-1, None]
    te_base = te_upper + w * (te_lower - te_upper)
    return np.vstack((airfoil, te_base))


# ============================================================================
# FAR-FIELD CIRCLE
# ============================================================================
def farfield_polyline(n_farfield=n_farfield):
    """Far-field circle, counter-clockwise."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_farfield, endpoint=False)
    return np.column_stack((x_offset + radius * np.cos(phi),
                            y_offset + radius * np.sin(phi)))


# ============================================================================
# RIBBON STL
# ============================================================================
def signed_area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def ribbon_facets(loop, outward_from_fluid_is_left):
    """Extrude a closed 2D polygon in z and split each quad into two triangles.

    Normals point out of the fluid domain (OpenFOAM convention for boundary
    faces). For a segment p0 -> p1, the right-hand normal is (dy, -dx).
    """
    facets = []
    n = len(loop)
    for i in range(n):
        p0, p1 = loop[i], loop[(i + 1) % n]
        a = np.array([p0[0], p0[1], z_min])
        b = np.array([p1[0], p1[1], z_min])
        c = np.array([p1[0], p1[1], z_max])
        d = np.array([p0[0], p0[1], z_max])
        tris = [(a, b, c), (a, c, d)]
        if outward_from_fluid_is_left:
            tris = [(a, c, b), (a, d, c)]
        for v0, v1, v2 in tris:
            nrm = np.cross(v1 - v0, v2 - v0)
            nrm /= np.linalg.norm(nrm)
            facets.append((nrm, v0, v1, v2))
    return facets


def write_solid(f, name, facets):
    f.write(f"solid {name}\n")
    for nrm, v0, v1, v2 in facets:
        f.write(f"  facet normal {nrm[0]:.9e} {nrm[1]:.9e} {nrm[2]:.9e}\n")
        f.write("    outer loop\n")
        for v in (v0, v1, v2):
            f.write(f"      vertex {v[0]:.12e} {v[1]:.12e} {v[2]:.12e}\n")
        f.write("    endloop\n")
        f.write("  endfacet\n")
    f.write(f"endsolid {name}\n")


def main():
    coord_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "oat15a_coordinates_c015.dat")
    stl_file = sys.argv[2] if len(sys.argv) > 2 else "oat15a_airfoil.stl"
    airfoil_loop = airfoil_polyline(coord_file)
    farfield_loop = farfield_polyline()
    te_upper, te_lower = airfoil_loop[n_airfoil], airfoil_loop[0]

    # Airfoil loop is clockwise (lower surface first, TE -> LE): the fluid lies to
    # the left of the travel direction, so the normal out of the fluid (into the
    # airfoil) is the right-hand normal. The far-field circle is counter-clockwise
    # with the fluid on its left, so its outward normal is again the right-hand one.
    assert signed_area(airfoil_loop) < 0.0, "airfoil loop should be clockwise"
    assert signed_area(farfield_loop) > 0.0, "far-field loop should be counter-clockwise"

    with open(stl_file, "w") as f:
        write_solid(f, "airfoil", ribbon_facets(airfoil_loop, False))
        write_solid(f, "farfield", ribbon_facets(farfield_loop, False))

    seg = np.hypot(*np.diff(np.vstack((airfoil_loop, airfoil_loop[:1])), axis=0).T)
    print(f"Wrote {stl_file}")
    print(f"  airfoil: {len(airfoil_loop)} segments, "
          f"length min/max = {seg.min():.3e}/{seg.max():.3e}, "
          f"TE thickness = {np.linalg.norm(te_upper - te_lower):.4e}")
    print(f"  farfield: {len(farfield_loop)} segments, radius = {radius}")


if __name__ == "__main__":
    main()
