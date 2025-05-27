#include "rb3d.h"

__global__ void macrovar(double *f, double *g, double *force_realx, double *force_realy, double *force_realz, 
                        double *rho, double *ux, double *uy, double *uz, double *phi) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LY + iy) * LX + ix;
    double rho_local = 0.0;
    double ux_local = 0.0;
    double uy_local = 0.0;
    double uz_local = 0.0;
    double phi_local = 0.0;

    #pragma unroll
    for (int ip = 0; ip < NPOP; ip++) {
        double f_val = f[ip * LXYZ + idx];
        double g_val = g[ip * LXYZ + idx];
        
        rho_local += f_val;
        ux_local  += d_cix[ip] * f_val;
        uy_local  += d_ciy[ip] * f_val;
        uz_local  += d_ciz[ip] * f_val;
        phi_local += g_val;
    }

    rho[idx] = rho_local;
    ux[idx]  = ux_local + 0.5 * force_realx[idx]; 
    uy[idx]  = uy_local + 0.5 * force_realy[idx];
    uz[idx]  = uz_local + 0.5 * force_realz[idx];
    phi[idx] = phi_local;
}

__global__ void forcing(double *force_realx, double *force_realy, double *force_realz, double *phi) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LY + iy) * LX + ix;

    force_realx[idx] = 0.0;  
    force_realy[idx] = d_grav0 * d_beta * (phi[idx] - d_t0); 
    force_realz[idx] = 0.0;  
}

__global__ void compute_uyT(double *uy, double *phi, double *uyT) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LY + iy) * LX + ix;
    
    uyT[idx] = uy[idx] * phi[idx];
}
