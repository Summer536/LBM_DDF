#include "rb3d.h"
#include "multi_gpu.h"

// 多GPU并行BGK碰撞kernel
__global__ void collision_BGK_parallel(double *f_local, double *force_realx_local, double *force_realy_local, double *force_realz_local, 
                                      double *rho_local, double *ux_local, double *uy_local, double *uz_local,
                                      int LY_local_with_halo) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_local = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ix >= LX || iy_local >= LY_local_with_halo || iz >= LZ) return;

    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    int local_size = LX * LY_local_with_halo * LZ;
    
    double fx9  = force_realx_local[idx_local];
    double fy9  = force_realy_local[idx_local];
    double fz9  = force_realz_local[idx_local];
    double rho9 = rho_local[idx_local];
    double u9   = ux_local[idx_local];
    double v9   = uy_local[idx_local];
    double w9   = uz_local[idx_local];

    double tau_xx = 0.0, tau_xy = 0.0, tau_xz = 0.0;
    double tau_yy = 0.0, tau_yz = 0.0, tau_zz = 0.0;

    #pragma unroll
    for (int ip = 0; ip < NPOP; ip++) {
        double RT = 1.0 / 3.0; 
        double eu = (d_cix[ip] * u9 + d_ciy[ip] * v9 + d_ciz[ip] * w9) / RT;
        double uv = (u9 * u9 + v9 * v9 + w9 * w9) / RT;
        double feq = d_tp[ip] * (rho9 + eu + 0.5 * (eu * eu - uv));

        double fchange = -(f_local[ip * local_size + idx_local] - feq) / d_tau;
        double feq_rho0 = d_tp[ip] * (1.0 + eu + 0.5 * (eu * eu - uv));
        fchange += (1.0 - 0.5 / d_tau) * fy9 * (d_ciy[ip] - v9) / RT * feq_rho0;
        f_local[ip * local_size + idx_local] += fchange;

        double fneq = f_local[ip * local_size + idx_local] - feq;
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

__global__ void collision_BGK_scalar_parallel(double *g_local, double *force_realx_local, double *force_realy_local, double *force_realz_local, 
                                            double *rho_local, double *ux_local, double *uy_local, double *uz_local, double *phi_local,
                                            int LY_local_with_halo) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_local = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ix >= LX || iy_local >= LY_local_with_halo || iz >= LZ) return;

    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    int local_size = LX * LY_local_with_halo * LZ;
    
    double phi9 = phi_local[idx_local];
    double u9   = ux_local[idx_local];
    double v9   = uy_local[idx_local];
    double w9   = uz_local[idx_local];
    double RT = 1.0 / 3.0;
    
    double g9[NPOP];

    #pragma unroll
    for (int ip = 0; ip < NPOP; ip++) {
        g9[ip] = g_local[ip * local_size + idx_local];
        double eu = (d_cix[ip] * u9 + d_ciy[ip] * v9 + d_ciz[ip] * w9) / RT;
        double uv = (u9 * u9 + v9 * v9 + w9 * w9) / RT;
        double geq = d_tp[ip] * phi9 * (1.0 + eu + 0.5 * (eu * eu - uv));
        double gchange = -(g9[ip] - geq) / d_tauc;
        g_local[ip * local_size + idx_local] += gchange;
    }
}

void collision_step_parallel(int num_gpus) {
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        GPU_Domain *domain = gpu_manager->domains[gpu_id];
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        
        dim3 block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);
        dim3 grid_size((LX + block_size.x - 1) / block_size.x,
                      (domain->LY_local_with_halo + block_size.y - 1) / block_size.y,
                      (LZ + block_size.z - 1) / block_size.z);
        
        collision_BGK_parallel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_f_local, domain->d_force_realx_local, domain->d_force_realy_local, domain->d_force_realz_local,
            domain->d_rho_local, domain->d_ux_local, domain->d_uy_local, domain->d_uz_local,
            domain->LY_local_with_halo);
        
        collision_BGK_scalar_parallel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_g_local, domain->d_force_realx_local, domain->d_force_realy_local, domain->d_force_realz_local,
            domain->d_rho_local, domain->d_ux_local, domain->d_uy_local, domain->d_uz_local, domain->d_phi_local,
            domain->LY_local_with_halo);
    }
}