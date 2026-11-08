// mc_core.h - per-cell Marching Cubes logic shared by the CUDA kernels and the CPU test harness.
// Same cube/edge conventions as the thesis (Appendix 1): corner bit values 1,2,4,8,16,32,64,128 for
// (x,y,z) (x+1,y,z) (x+1,y+1,z) (x,y+1,z) (x,y,z+1) (x+1,y,z+1) (x+1,y+1,z+1) (x,y+1,z+1);
// volume layout: unsigned char, x fastest, then y, then z (thesis reads the .raw file in that order).
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define MC_HD __host__ __device__
#else
#define MC_HD
#endif

struct MCParams {
    int   nx, ny;          // volume size in x,y (full)
    int   z0;              // global z index of first slice held in this buffer
    int   cellsZ;          // number of cell layers in this slab
    float level;           // thesis: mLevel = isovalue + 0.1
    int   bitIfGreater;    // 0: corner bit set if value < level (built-in Bourke table); 1: if value > level (thesis colvox.brw)
    float ox, oy, oz;      // origin (thesis: -n/2)
    float sx, sy, sz;      // voxel steps (thesis: 1)
};

struct MCFloat3 { float x, y, z; };

MC_HD inline void mc_corner_offset(int c, int& dx, int& dy, int& dz) {
    // corner c -> offset inside the cell
    static const int8_t O[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    dx = O[c][0]; dy = O[c][1]; dz = O[c][2];
}

MC_HD inline void mc_edge_corners(int e, int& a, int& b) {
    static const int8_t E[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
    a = E[e][0]; b = E[e][1];
}

// Reads the 8 corner values of cell (x,y,zl) (zl = z relative to slab buffer) and returns the case index.
MC_HD inline int mc_cube_index(const unsigned char* vol, const MCParams& p, int x, int y, int zl, float v[8]) {
    int idx = 0;
    for (int c = 0; c < 8; ++c) {
        int dx, dy, dz; mc_corner_offset(c, dx, dy, dz);
        size_t i = (size_t)(x + dx) + (size_t)p.nx * ((size_t)(y + dy) + (size_t)p.ny * (size_t)(zl + dz));
        v[c] = (float)vol[i];
        bool in = p.bitIfGreater ? (v[c] > p.level) : (v[c] < p.level);
        if (in) idx |= (1 << c);
    }
    return idx;
}

// Linear interpolation of the isosurface crossing on edge e (same formula as InterpolateVert() in the thesis).
MC_HD inline MCFloat3 mc_edge_vertex(const MCParams& p, int x, int y, int zg, const float v[8], int e) {
    int a, b; mc_edge_corners(e, a, b);
    int ax, ay, az, bx, by, bz;
    mc_corner_offset(a, ax, ay, az); mc_corner_offset(b, bx, by, bz);
    float d = v[a] - v[b];
    float t = (d > 1e-6f || d < -1e-6f) ? (v[a] - p.level) / d : 0.5f;
    float fx = (float)(x + ax) + t * (float)(bx - ax);
    float fy = (float)(y + ay) + t * (float)(by - ay);
    float fz = (float)(zg + az) + t * (float)(bz - az);
    MCFloat3 r = { p.ox + fx * p.sx, p.oy + fy * p.sy, p.oz + fz * p.sz };
    return r;
}
