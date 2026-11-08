// Gmsh geometry of the forward-facing step channel of Woodward and Colella:
// the rectangle [0, 3] x [0, 1] without the step [1, 3] x [0, 1/3] in its lower right corner.
//
// The mesh consists of quadrilaterals and is graded towards the step: the characteristic
// length at the nodes of the face and the corner of the step is `meshSize / refinement`, the
// length at nodes at the distance `transition` from the step and farther away is `meshSize`.
// Gmsh interpolates the lengths in between, so that the element size grows linearly from the
// step to the surrounding control nodes.
//
// Generate the Abaqus mesh file read by Trixi.jl with
//
//   gmsh -setnumber meshSize 0.0833 -setnumber refinement 8 -2 -o forward_facing_step.inp forward_facing_step.geo
//
// The boundary parts are the physical lines `inflow` (x = 0), `outflow` (x = 3), and `wall`
// (bottom, top, and the two faces of the step).

// command line parameters (gmsh -setnumber <name> <value>)
If(!Exists(meshSize))
  meshSize = 1.0 / 12.0;     // characteristic length far from the step
EndIf
If(!Exists(refinement))
  refinement = 8;            // size ratio between the far field and the step
EndIf
If(!Exists(transition))
  transition = 0.4;          // distance from the step at which the length is back to meshSize
EndIf

// Gmsh setup: quadrilaterals instead of the default triangles, node sets of the physical
// lines in the Abaqus file (used by Trixi.jl for the boundary names)
Mesh.RecombineAll = 1;
Mesh.RecombinationAlgorithm = 1;   // blossom: good quality quads
Mesh.Algorithm = 8;                // frontal-Delaunay for quads
Mesh.SaveAll = 1;
Mesh.SaveGroupsOfElements = 1;
Mesh.SaveGroupsOfNodes = 1;

clFar = meshSize;
clStep = meshSize / refinement;
d = transition;

// corners of the domain
Point(1) = {0, 0, 0, clFar};
Point(2) = {1 - d, 0, 0, clFar};        // control node on the bottom wall
Point(3) = {1, 0, 0, clStep};           // foot of the step
Point(4) = {1, 1.0/3.0, 0, clStep};     // corner of the step
Point(5) = {1 + d, 1.0/3.0, 0, clFar};  // control node on the top of the step
Point(6) = {3, 1.0/3.0, 0, clFar};
Point(7) = {3, 1, 0, clFar};
Point(8) = {0, 1, 0, clFar};

// interior control nodes at the distance `transition` from the step, which bound the refined
// region (embedded in the surface below)
Point(9)  = {1 - d, 1.0/3.0, 0, clFar};
Point(10) = {1 - d, 1.0/3.0 + d, 0, clFar};
Point(11) = {1, 1.0/3.0 + d, 0, clFar};
Point(12) = {1 + d, 1.0/3.0 + d, 0, clFar};

Line(1) = {1, 2};   // bottom wall, far part
Line(2) = {2, 3};   // bottom wall, refined part
Line(3) = {3, 4};   // face of the step
Line(4) = {4, 5};   // top of the step, refined part
Line(5) = {5, 6};   // top of the step, far part
Line(6) = {6, 7};   // outflow
Line(7) = {7, 8};   // top wall
Line(8) = {8, 1};   // inflow

Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};
Point{9, 10, 11, 12} In Surface{1};

Physical Line("inflow") = {8};
Physical Line("outflow") = {6};
Physical Line("wall") = {1, 2, 3, 4, 5, 7};
Physical Surface("domain") = {1};
