// mc_cuda.cu - GPU (CUDA) port of the parallel Marching Cubes from A. Onik, Harvard thesis (2008), Appendix 1.
//
// Thesis (MPI, master/worker): master splits the volume into Z-slabs (CalcNodeLoad), sends tables to the
//   workers, each worker reads its slab, loops over cells serially (GenerateTrianglesB), writes triangles
//   to a per-node .v3o file, and a master assembles them.
// Here (one GPU, any number of slabs): the Z-slab decomposition is kept (needed when the volume does not
//   fit in GPU memory; slabs overlap by one slice, like the thesis ranges), but the per-cell loop becomes
//   three data-parallel passes:
//     1. classify   : one thread per cell -> case index -> triangle count        (thesis step 1+2)
//     2. scan       : exclusive prefix sum of counts -> output offset of each cell (replaces the serial
//                     "NumberOfTriangles++" append and the PrevFound skipping trick)
//     3. generate   : one thread per active cell interpolates edge vertices and writes its triangles
//                     at its own offset                                           (thesis step 3)
//
// Build:  nvcc -O3 -std=c++17 mc_cuda.cu -o mc_cuda
// Run:    ./mc_cuda volume.raw nX nY nZ isovalue [--slab 64] [--out out.v3o] [--obj out.obj]
//                   [--topology colvox.brw]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "mc_core.h"
#include "mc_tables.h"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

// Table lives in constant memory (4 KB). Row layout: 16 entries, up to 5 triangles, -1 terminated.
__constant__ int8_t c_tri[256][16];

__global__ void classify_kernel(const unsigned char* vol, MCParams p, unsigned int* counts) {
    long long n = (long long)(p.nx - 1) * (p.ny - 1) * p.cellsZ;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int cx = p.nx - 1, cy = p.ny - 1;
    int x = (int)(id % cx);
    int y = (int)((id / cx) % cy);
    int zl = (int)(id / ((long long)cx * cy));
    float v[8];
    int idx = mc_cube_index(vol, p, x, y, zl, v);
    unsigned int nt = 0;
    if (idx != 0 && idx != 255) {
        while (nt < 5 && c_tri[idx][nt * 3] >= 0) ++nt;
    }
    counts[id] = nt;
}

__global__ void generate_kernel(const unsigned char* vol, MCParams p, const unsigned int* counts,
                                const unsigned int* offsets, MCFloat3* tris) {
    long long n = (long long)(p.nx - 1) * (p.ny - 1) * p.cellsZ;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    unsigned int nt = counts[id];
    if (nt == 0) return;
    int cx = p.nx - 1, cy = p.ny - 1;
    int x = (int)(id % cx);
    int y = (int)((id / cx) % cy);
    int zl = (int)(id / ((long long)cx * cy));
    float v[8];
    int idx = mc_cube_index(vol, p, x, y, zl, v);
    MCFloat3* out = tris + (size_t)offsets[id] * 3;
    for (unsigned int t = 0; t < nt; ++t)
        for (int k = 0; k < 3; ++k)
            out[t * 3 + k] = mc_edge_vertex(p, x, y, p.z0 + zl, v, c_tri[idx][t * 3 + k]);
}

// ---- optional: load the thesis topology file (colvox.brw) and convert it to the built-in layout ----
// File layout read by ReadTopology(): for j in 0..30, for i in 0..255: short -> TriangleTopology[i][j]
// ([i][0] = triangle count, then 3 one-based edge ids per triangle); then EdgeTopology (unused here).
// The thesis numbers edges 1..12 with edges 11 and 12 swapped relative to the Bourke table used by the
// built-in one (thesis 11 = corners 3-7, 12 = corners 2-6), so ids are remapped. Thesis convention is
// "corner bit set when value > level".
static bool load_colvox(const char* path, int8_t out[256][16]) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return false;
    static short T[256][31];
    for (int j = 0; j < 31; ++j)
        for (int i = 0; i < 256; ++i) { short s; if (fread(&s, sizeof(short), 1, fp) != 1) { fclose(fp); return false; } T[i][j] = s; }
    fclose(fp);
    for (int i = 0; i < 256; ++i) {
        for (int k = 0; k < 16; ++k) out[i][k] = -1;
        int nt = T[i][0];
        if (nt < 0 || nt > 5) { fprintf(stderr, "colvox: case %d has %d triangles (more than 5 not supported)\n", i, nt); return false; }
        for (int k = 0; k < nt * 3; ++k) {
            int e = T[i][1 + k];
            if (e < 1 || e > 12) return false;
            out[i][k] = (int8_t)(e == 11 ? 11 : (e == 12 ? 10 : e - 1));
        }
    }
    return true;
}

struct VoxInfT { int X_size, Y_size, Z_size; float Z_step, X_step, Y_step, CutLevel; int SmoothN; };  // as in the thesis

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s volume.raw nX nY nZ isovalue [--slab N] [--out f.v3o] [--obj f.obj] [--topology colvox.brw]\n", argv[0]); return 1; }
    const char* rawPath = argv[1];
    int nX = atoi(argv[2]), nY = atoi(argv[3]), nZ = atoi(argv[4]);
    float iso = (float)atof(argv[5]);
    int slab = 64; std::string outV3o = "out.v3o", outObj, topo;
    for (int i = 6; i < argc; ++i) {
        if (!strcmp(argv[i], "--slab") && i + 1 < argc) slab = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outV3o = argv[++i];
        else if (!strcmp(argv[i], "--obj") && i + 1 < argc) outObj = argv[++i];
        else if (!strcmp(argv[i], "--topology") && i + 1 < argc) topo = argv[++i];
    }
    if (nX < 2 || nY < 2 || nZ < 2) { fprintf(stderr, "volume too small\n"); return 1; }

    int8_t table[256][16]; memcpy(table, MC_TRI_TABLE, sizeof(table));
    int bitIfGreater = 0;
    if (!topo.empty()) {
        if (!load_colvox(topo.c_str(), table)) { fprintf(stderr, "cannot read topology %s\n", topo.c_str()); return 1; }
        bitIfGreater = 1;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_tri, table, sizeof(table)));

    FILE* fp = fopen(rawPath, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", rawPath); return 1; }

    MCParams p;
    p.nx = nX; p.ny = nY; p.level = iso + 0.1f;  // thesis: mLevel = limit + 0.1
    p.bitIfGreater = bitIfGreater;
    p.ox = -(float)nX / 2; p.oy = -(float)nY / 2; p.oz = -(float)nZ / 2;  // thesis SetVoxelDescription()
    p.sx = p.sy = p.sz = 1.0f;

    std::vector<MCFloat3> allTris;           // host copy of the result (thesis: per-node mesh files)
    std::vector<unsigned char> hostSlab;
    unsigned char* dVol = nullptr; unsigned int *dCounts = nullptr, *dOffsets = nullptr; MCFloat3* dTris = nullptr;
    size_t volCap = 0, cellCap = 0, triCap = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);

    const size_t sliceBytes = (size_t)nX * nY;
    for (int z0 = 0; z0 < nZ - 1; z0 += slab) {
        int cellsZ = std::min(slab, nZ - 1 - z0);            // cell layers in this slab
        int slices = cellsZ + 1;                              // slabs overlap by one slice (like CalcNodeLoad)
        hostSlab.resize(sliceBytes * slices);
        if (fseek(fp, (long)(sliceBytes * z0), SEEK_SET) != 0 || fread(hostSlab.data(), 1, hostSlab.size(), fp) != hostSlab.size()) {
            fprintf(stderr, "short read at slice %d\n", z0); return 1;
        }
        size_t cells = (size_t)(nX - 1) * (nY - 1) * cellsZ;
        if (hostSlab.size() > volCap) { cudaFree(dVol); CUDA_CHECK(cudaMalloc(&dVol, hostSlab.size())); volCap = hostSlab.size(); }
        if (cells > cellCap) { cudaFree(dCounts); cudaFree(dOffsets);
            CUDA_CHECK(cudaMalloc(&dCounts, cells * sizeof(unsigned))); CUDA_CHECK(cudaMalloc(&dOffsets, cells * sizeof(unsigned))); cellCap = cells; }
        CUDA_CHECK(cudaMemcpy(dVol, hostSlab.data(), hostSlab.size(), cudaMemcpyHostToDevice));
        p.z0 = z0; p.cellsZ = cellsZ;

        int threads = 256; long long blocks = ((long long)cells + threads - 1) / threads;
        classify_kernel<<<(unsigned)blocks, threads>>>(dVol, p, dCounts);
        CUDA_CHECK(cudaGetLastError());
        thrust::exclusive_scan(thrust::device, dCounts, dCounts + cells, dOffsets);   // pass 2
        unsigned lastOff = 0, lastCnt = 0;
        CUDA_CHECK(cudaMemcpy(&lastOff, dOffsets + cells - 1, sizeof(unsigned), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&lastCnt, dCounts + cells - 1, sizeof(unsigned), cudaMemcpyDeviceToHost));
        size_t nTri = (size_t)lastOff + lastCnt;
        if (nTri == 0) continue;
        if (nTri > triCap) { cudaFree(dTris); CUDA_CHECK(cudaMalloc(&dTris, nTri * 3 * sizeof(MCFloat3))); triCap = nTri; }
        generate_kernel<<<(unsigned)blocks, threads>>>(dVol, p, dCounts, dOffsets, dTris);
        CUDA_CHECK(cudaGetLastError());
        size_t old = allTris.size(); allTris.resize(old + nTri * 3);
        CUDA_CHECK(cudaMemcpy(allTris.data() + old, dTris, nTri * 3 * sizeof(MCFloat3), cudaMemcpyDeviceToHost));
        fprintf(stderr, "slab z=%d..%d: %zu triangles\n", z0, z0 + cellsZ, nTri);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    fclose(fp);
    size_t total = allTris.size() / 3;
    fprintf(stderr, "total triangles: %zu, GPU+transfer time: %.1f ms\n", total, ms);

    // .v3o output: int triangleCount, VoxInfT, then triangles (3 x float3 each) - same record layout as the thesis,
    // but with the count written once at the end (the thesis wrote only the first chunk's count and started at index 1).
    if (FILE* fo = fopen(outV3o.c_str(), "wb")) {
        int n = (int)total; VoxInfT vi = { nX, nY, nZ, 1.f, 1.f, 1.f, 100.f, 0 };
        fwrite(&n, sizeof(int), 1, fo); fwrite(&vi, sizeof(vi), 1, fo);
        fwrite(allTris.data(), sizeof(MCFloat3), allTris.size(), fo); fclose(fo);
    }
    if (!outObj.empty()) {
        if (FILE* fo = fopen(outObj.c_str(), "w")) {
            for (size_t i = 0; i < allTris.size(); ++i) fprintf(fo, "v %f %f %f\n", allTris[i].x, allTris[i].y, allTris[i].z);
            for (size_t t = 0; t < total; ++t) fprintf(fo, "f %zu %zu %zu\n", 3 * t + 1, 3 * t + 2, 3 * t + 3);
            fclose(fo);
        }
    }
    cudaFree(dVol); cudaFree(dCounts); cudaFree(dOffsets); cudaFree(dTris);
    return 0;
}
