# Vulkan Raytracing Test

Finished assignment for task 1

## Build

Same as in template https://github.com/SammaelA/SdfTaskTemplate

Clone this repo with its submodules:

    git clone https://github.com/AurumSN/vulkan-raytracing.git
    cd vulkan-raytracing
    git submodule update --init

Install SDL and Vulkan.

Build the executable:

    cmake -B build && cmake --build build

## Execute
    
    ./render [input file name] {-o [output file name]} {-sph [x] [y] [z] [r]} {-s [size]} {-d [depth]}

* -o --- choose output file
* -sph --- create a "cutting" sphere at (x, y, z) with radius r
* -s --- choose size of the SDF Grid
* -d --- choose depth of the SDF Octree

## Contents

An application that uses Vulkan to render an image with a compute shader and render it to the window.
The image is created using Raytracing and Spheretracing techniques:
* A. Can render some primitives and OBJ Meshes with BVH optimisations, shadows and reflections;
    - [x] A1
    - [x] A2
    - [x] A3
    - [x] A4
    - [x] A5
* B. Can render SDF Grids, can generate SDF Grid from mesh, can generate SDF Grid from other SDF Grid cutting out a sphere;
    - [x] B1
    - [x] B2
    - [x] B3
* C. Can render SDF Octrees, can generate SDF Octree from mesh, can generate SDF Octree from other SDF Octree cutting out a sphere;
    - [x] C1
    - [x] C2
    - [x] C3
* D. Supports ambient occlusion, uses Fast Sweep to generate SDF Grids, supports analytic method and Newton's method of finding roots to a cubic equation.
    - [x] D1
    - [x] D2
    - [x] D3
    - [x] D4