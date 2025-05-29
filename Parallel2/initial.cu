#include "rb3d.h"
#include "multi_gpu.h"

void copy_global_to_local_macrovar(double *h_rho_global, double *h_ux_global, double *h_uy_global, 
                                  double *h_uz_global, double *h_phi_global, GPU_Domain *domain);
void copy_global_to_local_distributions(double *h_f_global, double *h_g_global, GPU_Domain *domain);

void initialize_parallel(int num_gpus) {
    if (multi_gpu_init(num_gpus) != 0) {
        fprintf(stderr, "Failed to initialize multi-GPU system\n");
        exit(EXIT_FAILURE);
    }
    
    setup_gpu_domains();
    
    check_p2p_capability();
    enable_p2p_access();
    
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        allocate_local_memory(gpu_manager->domains[gpu_id]);
    }
    
    initialize_constants();
    
    if (continue_step == 0) {
        initialize_arrays_parallel(num_gpus);
    } else {
        initialize_from_file_parallel(num_gpus);
    }
    
    printf("Multi-GPU initialization completed successfully\n");
}

void initialize_constants() {
    //                 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26 
    int h_cix[NPOP] = {0,  1,  0, -1,  0,  0,  0,  1, -1, -1,  1,  1,  0, -1,  0,  1,  0, -1,  0,  1, -1, -1,  1,  1, -1, -1,  1};
    int h_ciy[NPOP] = {0,  0,  1,  0, -1,  0,  0,  1,  1, -1, -1,  0,  1,  0, -1,  0,  1,  0, -1,  1,  1, -1, -1,  1,  1, -1, -1};
    int h_ciz[NPOP] = {0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1};
     
    double h_tp[NPOP];

    #pragma unroll
    for (int i = 0; i < NPOP; i++) {
        if      (i == 0)  h_tp[i] = 8.0/27.0;
        else if (i <= 6)  h_tp[i] = 2.0/27.0;
        else if (i <= 18) h_tp[i] = 1.0/54.0;
        else              h_tp[i] = 1.0/216.0;
    }
    
    for (int gpu_id = 0; gpu_id < gpu_manager->num_gpus; gpu_id++) {
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cix,   h_cix, NPOP * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_ciy,   h_ciy, NPOP * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_ciz,   h_ciz, NPOP * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tp,    h_tp,  NPOP * sizeof(double)));

        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tau,   &tau,         sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tauc,  &tauc,        sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_grav0, &grav0,       sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_beta,  &beta,        sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tHot,  &tHot,        sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_tCold, &tCold,       sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_t0,    &t0,          sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_diff,  &diff,        sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_visc,  &visc,        sizeof(double)));
    }
}

__global__ void init_arrays_parallel(double *f_local, double *g_local, double *rho_local, double *ux_local, double *uy_local, double *uz_local,
                                    double *phi_local, double *force_realx_local, double *force_realy_local, double *force_realz_local,
                                    int LY_local_with_halo, int y_start_global) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_local = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ix >= LX || iy_local >= LY_local_with_halo || iz >= LZ) return;

    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    int local_size = LX * LY_local_with_halo * LZ;
    
    int iy_global = get_global_y_from_local(iy_local, y_start_global);
    
    double initial_rho = 1.0;
    double initial_ux = 0.0;
    double initial_uy = 0.0; 
    double initial_uz = 0.0;
    
    double initial_phi = d_tHot - (d_tHot - d_tCold) * iy_global / (LY - 1.0);
    
    rho_local[idx_local] = initial_rho;
    ux_local[idx_local] = initial_ux;
    uy_local[idx_local] = initial_uy;
    uz_local[idx_local] = initial_uz;
    phi_local[idx_local] = initial_phi;

    force_realx_local[idx_local] = 0.0;
    force_realy_local[idx_local] = d_grav0 * d_beta * (initial_phi - d_t0);
    force_realz_local[idx_local] = 0.0;

    double RT = 1.0 / 3.0;
    double uv = (initial_ux * initial_ux + initial_uy * initial_uy + initial_uz * initial_uz) / RT;
    
    #pragma unroll
    for (int ip = 0; ip < NPOP; ip++) {
        double eu = (d_cix[ip] * initial_ux + d_ciy[ip] * initial_uy + d_ciz[ip] * initial_uz) / RT;
        
        double feq = d_tp[ip] * (initial_rho + eu + 0.5 * (eu * eu - uv));
        f_local[ip * local_size + idx_local] = feq;
        
        double geq = d_tp[ip] * initial_phi * (1.0 + eu + 0.5 * (eu * eu - uv));
        g_local[ip * local_size + idx_local] = geq;
    }
}

void initialize_arrays_parallel(int num_gpus) {
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        GPU_Domain *domain = gpu_manager->domains[gpu_id];
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        
        dim3 block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);
        dim3 grid_size((LX + block_size.x - 1) / block_size.x,
                      (domain->LY_local_with_halo + block_size.y - 1) / block_size.y,
                      (LZ + block_size.z - 1) / block_size.z);
        
        init_arrays_parallel<<<grid_size, block_size>>>(
            domain->d_f_local, domain->d_g_local, 
            domain->d_rho_local, domain->d_ux_local, domain->d_uy_local, domain->d_uz_local,
            domain->d_phi_local, domain->d_force_realx_local, domain->d_force_realy_local, domain->d_force_realz_local,
            domain->LY_local_with_halo, domain->y_start_global);
        
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}

void initialize_from_file_parallel(int num_gpus) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    char data_filename[512];
    snprintf(data_filename, sizeof(data_filename), "%s/%09d.dat", dirname, continue_step);

    char fg_filename[512];
    snprintf(fg_filename, sizeof(fg_filename), "%s/fg%09d.dat", dirname, continue_step);

    double *h_rho_global = (double*)malloc(LXYZ * sizeof(double));
    double *h_ux_global = (double*)malloc(LXYZ * sizeof(double));
    double *h_uy_global = (double*)malloc(LXYZ * sizeof(double));
    double *h_uz_global = (double*)malloc(LXYZ * sizeof(double));
    double *h_phi_global = (double*)malloc(LXYZ * sizeof(double));
    double *h_f_global = (double*)malloc(NPOP * LXYZ * sizeof(double));
    double *h_g_global = (double*)malloc(NPOP * LXYZ * sizeof(double));

    if (!h_rho_global || !h_ux_global || !h_uy_global || !h_uz_global || 
        !h_phi_global || !h_f_global || !h_g_global) {
        fprintf(stderr, "Failed to allocate memory for global arrays\n");
        exit(EXIT_FAILURE);
    }

    FILE *f_data = fopen(data_filename, "rb");
    if (f_data == NULL) {
        fprintf(stderr, "Cannot open file: %s\n", data_filename);
        exit(EXIT_FAILURE);
    }

    if (fread(h_rho_global, sizeof(double), LXYZ, f_data) != LXYZ ||
        fread(h_ux_global, sizeof(double), LXYZ, f_data) != LXYZ ||
        fread(h_uy_global, sizeof(double), LXYZ, f_data) != LXYZ ||
        fread(h_uz_global, sizeof(double), LXYZ, f_data) != LXYZ ||
        fread(h_phi_global, sizeof(double), LXYZ, f_data) != LXYZ) {
        fprintf(stderr, "Error reading macroscopic variables from %s\n", data_filename);
        exit(EXIT_FAILURE);
    }
    fclose(f_data);

    FILE *f_fg = fopen(fg_filename, "rb");
    if (f_fg == NULL) {
        fprintf(stderr, "Cannot open file: %s\n", fg_filename);
        exit(EXIT_FAILURE);
    }

    if (fread(h_f_global, sizeof(double), NPOP * LXYZ, f_fg) != NPOP * LXYZ ||
        fread(h_g_global, sizeof(double), NPOP * LXYZ, f_fg) != NPOP * LXYZ) {
        fprintf(stderr, "Error reading distribution functions from %s\n", fg_filename);
        exit(EXIT_FAILURE);
    }
    fclose(f_fg);

    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        GPU_Domain *domain = gpu_manager->domains[gpu_id];
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        
        copy_global_to_local_macrovar(h_rho_global, h_ux_global, h_uy_global, h_uz_global, h_phi_global, domain);
        
        copy_global_to_local_distributions(h_f_global, h_g_global, domain);
    }

    free(h_rho_global); free(h_ux_global); free(h_uy_global); 
    free(h_uz_global); free(h_phi_global); free(h_f_global); free(h_g_global);
}

void copy_global_to_local_macrovar(double *h_rho_global, double *h_ux_global, double *h_uy_global, 
                                  double *h_uz_global, double *h_phi_global, GPU_Domain *domain) {
    size_t local_field_size = domain->local_size * sizeof(double);
    
    for (int iz = 0; iz < LZ; iz++) {
        for (int iy_local = 0; iy_local < domain->LY_local_with_halo; iy_local++) {
            for (int ix = 0; ix < LX; ix++) {
                int iy_global = get_global_y_from_local(iy_local, domain->y_start_global);
                int idx_local = get_local_index(ix, iy_local, iz, domain->LY_local_with_halo);
                
                if (iy_global >= 0 && iy_global < LY) {
                    int idx_global = (iz * LY + iy_global) * LX + ix;
                    domain->h_rho_local[idx_local] = h_rho_global[idx_global];
                    domain->h_ux_local[idx_local] = h_ux_global[idx_global];
                    domain->h_uy_local[idx_local] = h_uy_global[idx_global];
                    domain->h_uz_local[idx_local] = h_uz_global[idx_global];
                    domain->h_phi_local[idx_local] = h_phi_global[idx_global];
                } else {
                    domain->h_rho_local[idx_local] = 1.0;
                    domain->h_ux_local[idx_local] = 0.0;
                    domain->h_uy_local[idx_local] = 0.0;
                    domain->h_uz_local[idx_local] = 0.0;
                    domain->h_phi_local[idx_local] = (iy_global < 0) ? tHot : tCold;
                }
            }
        }
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(domain->d_rho_local, domain->h_rho_local, local_field_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(domain->d_ux_local, domain->h_ux_local, local_field_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(domain->d_uy_local, domain->h_uy_local, local_field_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(domain->d_uz_local, domain->h_uz_local, local_field_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(domain->d_phi_local, domain->h_phi_local, local_field_size, cudaMemcpyHostToDevice));
}

void copy_global_to_local_distributions(double *h_f_global, double *h_g_global, GPU_Domain *domain) {
    size_t local_dist_size = NPOP * domain->local_size * sizeof(double);
    
    double *h_f_local = (double*)malloc(local_dist_size);
    double *h_g_local = (double*)malloc(local_dist_size);
    
    if (!h_f_local || !h_g_local) {
        fprintf(stderr, "Failed to allocate temporary local distribution arrays\n");
        exit(EXIT_FAILURE);
    }
    
    for (int ip = 0; ip < NPOP; ip++) {
        for (int iz = 0; iz < LZ; iz++) {
            for (int iy_local = 0; iy_local < domain->LY_local_with_halo; iy_local++) {
                for (int ix = 0; ix < LX; ix++) {
                    int iy_global = get_global_y_from_local(iy_local, domain->y_start_global);
                    int idx_local = get_local_index(ix, iy_local, iz, domain->LY_local_with_halo);
                    
                    if (iy_global >= 0 && iy_global < LY) {
                        int idx_global = (iz * LY + iy_global) * LX + ix;
                        h_f_local[ip * domain->local_size + idx_local] = h_f_global[ip * LXYZ + idx_global];
                        h_g_local[ip * domain->local_size + idx_local] = h_g_global[ip * LXYZ + idx_global];
                    } else {
                        double rho_val = 1.0;
                        double ux_val = 0.0, uy_val = 0.0, uz_val = 0.0;
                        double phi_val = (iy_global < 0) ? tHot : tCold;

                        int h_cix[NPOP] = {0,  1,  0, -1,  0,  0,  0,  1, -1, -1,  1,  1,  0, -1,  0,  1,  0, -1,  0,  1, -1, -1,  1,  1, -1, -1,  1};
                        int h_ciy[NPOP] = {0,  0,  1,  0, -1,  0,  0,  1,  1, -1, -1,  0,  1,  0, -1,  0,  1,  0, -1,  1,  1, -1, -1,  1,  1, -1, -1};
                        int h_ciz[NPOP] = {0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1};
                        double h_tp_val = (ip == 0) ? 8.0/27.0 : 
                                         (ip <= 6) ? 2.0/27.0 : 
                                         (ip <= 18) ? 1.0/54.0 : 1.0/216.0;
                        
                        double RT = 1.0 / 3.0;
                        double eu = (h_cix[ip] * ux_val + h_ciy[ip] * uy_val + h_ciz[ip] * uz_val) / RT;
                        double uv = (ux_val * ux_val + uy_val * uy_val + uz_val * uz_val) / RT;
                        
                        h_f_local[ip * domain->local_size + idx_local] = h_tp_val * (rho_val + eu + 0.5 * (eu * eu - uv));
                        h_g_local[ip * domain->local_size + idx_local] = h_tp_val * phi_val * (1.0 + eu + 0.5 * (eu * eu - uv));
                    }
                }
            }
        }
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(domain->d_f_local, h_f_local, local_dist_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(domain->d_g_local, h_g_local, local_dist_size, cudaMemcpyHostToDevice));
    
    free(h_f_local);
    free(h_g_local);
}

void initialize() {
    initialize_parallel(1);
}