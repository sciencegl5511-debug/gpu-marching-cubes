# GPU (CUDA) Marching Cubes — port of the MPI implementation in A. Onik's thesis (Appendix 1)

## Files
- `mc_cuda.cu`   host code + 2 kernels (classify, generate) + Thrust scan; CLI like the thesis program
- `mc_core.h`    per-cell logic (cube index, edge interpolation) — shared by GPU and CPU test
- `mc_tables.h`  classic Lorensen/Bourke 256-case table (checked: edge sets match for all 256 cases)
- `cpu_test.cpp` CPU emulation of the same three passes on a synthetic sphere (no GPU needed)

## Build / run (needs an NVIDIA GPU + CUDA Toolkit)
    nvcc -O3 -std=c++17 mc_cuda.cu -o mc_cuda
    ./mc_cuda stack_300x328x100.raw 300 328 100 100 --slab 64 --out out.v3o --obj out.obj
    g++ -O2 -std=c++17 cpu_test.cpp -o cpu_test && ./cpu_test      # logic check, any machine

Arguments match the thesis (`raw nX nY nZ isovalue`); raw file = unsigned 8-bit, x fastest, then y, then z.

## How the MPI design maps to the GPU
| Thesis (MPI)                                   | GPU version                                              |
|------------------------------------------------|----------------------------------------------------------|
| CalcNodeLoad: Z-slabs, one per node            | Z-slabs processed one after another (1-slice overlap)    |
| MPI_Send of topology tables to workers         | table copied once to GPU constant memory                 |
| serial per-cell loop (GenerateTrianglesB)      | one thread per cell                                      |
| step 1+2: active cell / case index             | `classify_kernel` (case index -> triangle count)         |
| NumberOfTriangles++ (serial append), PrevFound | exclusive prefix sum of counts -> per-cell output offset |
| step 3: InterpolateVert + SetTriangle          | `generate_kernel` writes at its own offset               |
| per-node .v3o files, master assembles          | one .v3o (same record layout) and optional .obj          |

Multiple GPUs: give each GPU a range of slabs, exactly like the MPI node ranges.

## Differences from the thesis code (deliberate)
1. Case table: your `colvox.brw` is not in the PDF, so a standard Bourke table is built in (same corner and
   edge numbering as the thesis, except thesis edges 11/12 are swapped vs Bourke). `--topology colvox.brw` loads
   your file instead (remaps edges 11/12; uses the thesis rule "corner bit set when value > level").
2. Thesis loops `y` from 1 and so skips the y=0 row of cells; here y starts at 0.
3. `SaveTriToFile` wrote triangles from index 1 (skipping triangle 0) and wrote the count only for the first
   chunk; here all triangles are written and the count is written once.
4. Level = isovalue + 0.1 (as in the thesis) so equality never occurs.
5. 32-bit offsets: keep slab cells below ~800M (cells x up to 5 triangles must fit in 32 bits).

## What was and was not tested
- Tested (CPU, same shared code): sphere r=20 on a 64^3 volume -> area 5018 vs 5027 exact; vertex radius within 0.08 voxel;
  identical triangle count with 17-slice slabs and one slab; every edge shared by exactly 2 triangles (watertight).
- NOT tested: the .cu file itself (no GPU or nvcc where it was written). Expect to fix small compile issues.
- The classic table has the known ambiguous-face cases (rare tiny holes), as any original-Lorensen implementation does.
