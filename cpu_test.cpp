// CPU emulation of the 3 GPU passes using the SAME mc_core.h + mc_tables.h, to validate logic without a GPU.
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include "mc_core.h"
#include "mc_tables.h"
int main() {
    const int N = 64; const float R = 20.0f;
    std::vector<unsigned char> vol((size_t)N*N*N);
    // value = clamp(128 + 8*(R - dist)) : smooth ramp, 128 at the sphere surface, higher inside
    for (int z=0;z<N;++z) for (int y=0;y<N;++y) for (int x=0;x<N;++x) {
        float d = sqrtf((x-31.5f)*(x-31.5f)+(y-31.5f)*(y-31.5f)+(z-31.5f)*(z-31.5f));
        float v = 128.f + 8.f*(R-d); v = v<0?0:(v>255?255:v);
        vol[(size_t)x + (size_t)N*(y + (size_t)N*z)] = (unsigned char)(v+0.5f);
    }
    MCParams p; p.nx=N; p.ny=N; p.level=128.f+0.1f; p.bitIfGreater=0;   // built-in table: bit set when value < level
    p.ox=p.oy=p.oz=-(float)N/2; p.sx=p.sy=p.sz=1.f;
    int slab = 17;                                      // deliberately not a divisor of N-1 to test slab edges
    std::vector<MCFloat3> tris;
    for (int z0=0; z0<N-1; z0+=slab) {
        int cz = std::min(slab, N-1-z0); p.z0=z0; p.cellsZ=cz;
        const unsigned char* base = vol.data() + (size_t)N*N*z0;      // slab buffer = slices z0..z0+cz
        size_t cells=(size_t)(N-1)*(N-1)*cz; std::vector<unsigned> cnt(cells), off(cells);
        for (size_t id=0; id<cells; ++id) {                           // pass 1: classify
            int x=id%(N-1), y=(id/(N-1))%(N-1), zl=id/((size_t)(N-1)*(N-1)); float v[8];
            int idx=mc_cube_index(base,p,x,y,zl,v); unsigned nt=0;
            if (idx!=0&&idx!=255) while(nt<5 && MC_TRI_TABLE[idx][nt*3]>=0) ++nt;
            cnt[id]=nt; }
        unsigned run=0; for (size_t i=0;i<cells;++i){ off[i]=run; run+=cnt[i]; }   // pass 2: exclusive scan
        size_t old=tris.size(); tris.resize(old+(size_t)run*3);
        for (size_t id=0; id<cells; ++id) { if(!cnt[id]) continue;     // pass 3: generate
            int x=id%(N-1), y=(id/(N-1))%(N-1), zl=id/((size_t)(N-1)*(N-1)); float v[8];
            int idx=mc_cube_index(base,p,x,y,zl,v);
            for (unsigned t=0;t<cnt[id];++t) for(int k=0;k<3;++k)
                tris[old+(size_t)(off[id]+t)*3+k]=mc_edge_vertex(p,x,y,z0+zl,v,MC_TRI_TABLE[idx][t*3+k]); }
    }
    double area=0, rmin=1e9, rmax=0; 
    for (size_t t=0;t<tris.size()/3;++t) {
        MCFloat3 a=tris[3*t],b=tris[3*t+1],c=tris[3*t+2];
        double ux=b.x-a.x,uy=b.y-a.y,uz=b.z-a.z,vx=c.x-a.x,vy=c.y-a.y,vz=c.z-a.z;
        double cx=uy*vz-uz*vy,cy=uz*vx-ux*vz,cz=ux*vy-uy*vx; area+=0.5*sqrt(cx*cx+cy*cy+cz*cz);
        for (int k=0;k<3;++k){ MCFloat3 q=tris[3*t+k]; double r=sqrt((q.x+0.5)*(q.x+0.5)+(q.y+0.5)*(q.y+0.5)+(q.z+0.5)*(q.z+0.5)); rmin=std::min(rmin,r); rmax=std::max(rmax,r);} }
    printf("triangles=%zu  area=%.1f (sphere 4*pi*R^2=%.1f)  vertex radius range [%.3f, %.3f] (R=%.1f)\n",
           tris.size()/3, area, 4*M_PI*R*R, rmin, rmax, R);
    // edge-manifold check: every undirected edge should be shared by exactly 2 triangles (closed surface)
    return 0;
}
