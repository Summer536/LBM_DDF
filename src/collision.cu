#include "rb3d.h"

__global__ void collision_BGK(double *f, double *force_realx, double *force_realy, double *force_realz, 
                             double *rho, double *ux, double *uy, double *uz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LY + iy) * LX + ix;
    double fx9  = force_realx[idx];
    double fy9  = force_realy[idx];
    double fz9  = force_realz[idx];
    double rho9 = rho[idx];
    double u9   = ux[idx];
    double v9   = uy[idx];
    double w9   = uz[idx];

    double tau_xx = 0.0, tau_xy = 0.0, tau_xz = 0.0;
    double tau_yy = 0.0, tau_yz = 0.0, tau_zz = 0.0;

    #pragma unroll
    for (int ip = 0; ip < NPOP; ip++) {
        double RT = 1.0 / 3.0; 
        double eu = (d_cix[ip] * u9 + d_ciy[ip] * v9 + d_ciz[ip] * w9) / RT;
        double uv = (u9 * u9 + v9 * v9 + w9 * w9) / RT;
        double feq = d_tp[ip] * (rho9 + eu + 0.5 * (eu * eu - uv));

        double fchange = -(f[ip * LXYZ + idx] - feq) / d_tau;
        double feq_rho0 = d_tp[ip] * (1.0 + eu + 0.5 * (eu * eu - uv));
        fchange += (1.0 - 0.5 / d_tau) * fy9 * (d_ciy[ip] - v9) / RT * feq_rho0;
        f[ip * LXYZ + idx] += fchange;

        double fneq = f[ip * LXYZ + idx] - feq;
        double cidotF = d_cix[ip]*fx9 + d_ciy[ip]*fy9 + d_ciz[ip]*fz9;
        double bu = fx9*u9 + fy9*v9 + fz9*w9;
        double feq_term = 0.5 * (cidotF - bu) * feq / RT;

        tau_xx += d_cix[ip] * d_cix[ip] * (fneq + feq_term);
        tau_xy += d_cix[ip] * d_ciy[ip] * (fneq + feq_term);
        tau_xz += d_cix[ip] * d_ciz[ip] * (fneq + feq_term);
        tau_yy += d_ciy[ip] * d_ciy[ip] * (fneq + feq_term);
        tau_yz += d_ciy[ip] * d_ciz[ip] * (fneq + feq_term);
        tau_zz += d_ciz[ip] * d_ciz[ip] * (fneq + feq_term);
    }

    double coeff = -(1.0 - 0.5/d_tau);
    tau_xx *= coeff; tau_xy *= coeff; tau_xz *= coeff;
    tau_yy *= coeff; tau_yz *= coeff; tau_zz *= coeff;
}

__global__ void collision_BGK_scalar(double *g, double *force_realx, double *force_realy, double *force_realz, 
                                    double *rho, double *ux, double *uy, double *uz, double *phi) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LY + iy) * LX + ix;
    double phi9 = phi[idx];
    double u9   = ux[idx];
    double v9   = uy[idx];
    double w9   = uz[idx];
    double RT = 1.0 / 3.0;
    
    // 共享内存使用有问题？？？？！！！！！
    // extern __shared__ double shared_mem[];
    // double *g9 = shared_mem;
    // double *geqnew = &shared_mem[NPOP];

    // double g9[NPOP], geqnew[NPOP];
    double g9[NPOP];

    #pragma unroll
    for (int ip = 0; ip < NPOP; ip++) {
        g9[ip] = g[ip * LXYZ + idx];
        double eu = (d_cix[ip] * u9 + d_ciy[ip] * v9 + d_ciz[ip] * w9) / RT;
        double uv = (u9 * u9 + v9 * v9 + w9 * w9) / RT;
        double geq = d_tp[ip] * phi9 * (1.0 + eu + 0.5 * (eu * eu - uv));
        double gchange = -(g9[ip] - geq) / d_tauc;
        g[ip * LXYZ + idx] += gchange;
      
        // if (ip == 0) {
        //     geqnew[ip] = geq - (1.0 - 8.0/27.0) * phi9 * rho9 / 1.0;  
        // } else {
        //     geqnew[ip] = geq + d_tp[ip] * phi9 * rho9 / 1.0;
        // }
    }

    // double gneqx = 0.0, gneqy = 0.0, gneqz = 0.0;
    // for (int ip = 0; ip < NPOP; ip++) {
    //     gneqx += d_cix[ip] * (g9[ip] - geqnew[ip]);
    //     gneqy += d_ciy[ip] * (g9[ip] - geqnew[ip]);
    //     gneqz += d_ciz[ip] * (g9[ip] - geqnew[ip]);
    // }

    // double Ralph_coeff = (1.0 - 1.0 / (2.0 * d_tauc));
    // for (int ip = 0; ip < NPOP; ip++) {
    //     double eb = d_cix[ip] * fx9 + d_ciy[ip] * fy9 + d_ciz[ip] * fz9;
    //     double gneq = d_cix[ip] * gneqx + d_ciy[ip] * gneqy + d_ciz[ip] * gneqz;
    //     double gradt = -(2.0 * gneq + eb * phi9) / (rho9 / 3.0 + 2.0 * d_tauc / 3.0);
    //     double Ralph = d_tp[ip] * Ralph_coeff * (gradt * rho9 / 1.0 + 3.0 * phi9 * eb);
    //     double gchange = -(g9[ip] - geqnew[ip]) / d_tauc + Ralph;
    //     g[ip * LXYZ + idx] += gchange;
    // }

    // double p = rho9 * RT;
    // double denominator = 2.0 * d_tauc * RT + p;
    
    // Tx[idx] = -(2.0 * gneqx + phi9 * fx9) / denominator;
    // Ty[idx] = -(2.0 * gneqy + phi9 * fy9) / denominator;
    // Tz[idx] = -(2.0 * gneqz + phi9 * fz9) / denominator;
}