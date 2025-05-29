#include "rb3d.h"
#include "multi_gpu.h"

__global__ void macrovar_parallel(double *f_local, double *g_local, double *force_realx_local, double *force_realy_local, double *force_realz_local, 
                                 double *rho_local, double *ux_local, double *uy_local, double *uz_local, double *phi_local,
                                 int LY_local_with_halo) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_local = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ix >= LX || iy_local >= LY_local_with_halo || iz >= LZ) return;

    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    int local_size = LX * LY_local_with_halo * LZ;
    
    double rho_val = 0.0;
    double ux_val = 0.0;
    double uy_val = 0.0;
    double uz_val = 0.0;
    double phi_val = 0.0;

    #pragma unroll
    for (int ip = 0; ip < NPOP; ip++) {
        double f_val = f_local[ip * local_size + idx_local];
        double g_val = g_local[ip * local_size + idx_local];
        
        rho_val += f_val;
        ux_val  += d_cix[ip] * f_val;
        uy_val  += d_ciy[ip] * f_val;
        uz_val  += d_ciz[ip] * f_val;
        phi_val += g_val;
    }

    rho_local[idx_local] = rho_val;
    ux_local[idx_local]  = ux_val + 0.5 * force_realx_local[idx_local]; 
    uy_local[idx_local]  = uy_val + 0.5 * force_realy_local[idx_local];
    uz_local[idx_local]  = uz_val + 0.5 * force_realz_local[idx_local];
    phi_local[idx_local] = phi_val;
}

__global__ void forcing_parallel(double *force_realx_local, double *force_realy_local, double *force_realz_local, double *phi_local,
                                int LY_local_with_halo) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_local = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ix >= LX || iy_local >= LY_local_with_halo || iz >= LZ) return;

    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);

    force_realx_local[idx_local] = 0.0;  
    force_realy_local[idx_local] = d_grav0 * d_beta * (phi_local[idx_local] - d_t0); 
    force_realz_local[idx_local] = 0.0;  
}

__global__ void compute_uyT_parallel(double *uy_local, double *phi_local, double *uyT_local, int LY_local_with_halo) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_local = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ix >= LX || iy_local >= LY_local_with_halo || iz >= LZ) return;

    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    
    uyT_local[idx_local] = uy_local[idx_local] * phi_local[idx_local];
}

void macrovar_step_parallel(int num_gpus) {
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        GPU_Domain *domain = gpu_manager->domains[gpu_id];
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        
        dim3 block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);
        dim3 grid_size((LX + block_size.x - 1) / block_size.x,
                      (domain->LY_local_with_halo + block_size.y - 1) / block_size.y,
                      (LZ + block_size.z - 1) / block_size.z);
        
        macrovar_parallel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_f_local, domain->d_g_local, 
            domain->d_force_realx_local, domain->d_force_realy_local, domain->d_force_realz_local,
            domain->d_rho_local, domain->d_ux_local, domain->d_uy_local, domain->d_uz_local, domain->d_phi_local,
            domain->LY_local_with_halo);
        
        forcing_parallel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_force_realx_local, domain->d_force_realy_local, domain->d_force_realz_local, domain->d_phi_local,
            domain->LY_local_with_halo);
    }
}
