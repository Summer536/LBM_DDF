#include "rb3d.h"

void initialize() {
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_f,      NPOP * LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_g,      NPOP * LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_f_temp, NPOP * LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_g_temp, NPOP * LXYZ * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_rho,           LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ux,            LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_uy,            LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_uz,            LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_phi,           LXYZ * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_force_realx,   LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_force_realy,   LXYZ * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_force_realz,   LXYZ * sizeof(double)));
    
    // Allocate host memory with error checking
    h_rho = (double*)malloc(LXYZ * sizeof(double));
    if (h_rho == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_rho\n");
        exit(EXIT_FAILURE);
    }
    
    h_ux = (double*)malloc(LXYZ * sizeof(double));
    if (h_ux == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_ux\n");
        free(h_rho);
        exit(EXIT_FAILURE);
    }
    
    h_uy = (double*)malloc(LXYZ * sizeof(double));
    if (h_uy == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_uy\n");
        free(h_rho); free(h_ux);
        exit(EXIT_FAILURE);
    }
    
    h_uz = (double*)malloc(LXYZ * sizeof(double));
    if (h_uz == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_uz\n");
        free(h_rho); free(h_ux); free(h_uy);
        exit(EXIT_FAILURE);
    }
    
    h_phi = (double*)malloc(LXYZ * sizeof(double));
    if (h_phi == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_phi\n");
        free(h_rho); free(h_ux); free(h_uy); free(h_uz);
        exit(EXIT_FAILURE);
    }
    
    h_f = (double*)malloc(NPOP * LXYZ * sizeof(double));
    if (h_f == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_f\n");
        free(h_rho); free(h_ux); free(h_uy); free(h_uz); free(h_phi);
        exit(EXIT_FAILURE);
    }
    
    h_g = (double*)malloc(NPOP * LXYZ * sizeof(double));
    if (h_g == NULL) {
        fprintf(stderr, "Failed to allocate host memory for h_g\n");
        free(h_rho); free(h_ux); free(h_uy); free(h_uz); free(h_phi); free(h_f);
        exit(EXIT_FAILURE);
    }

    // Initialize constant memory
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
    
    // Copy parameters to constant memory
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

    // Initialize device arrays
    dim3 grid((LX + BLOCK_X - 1) / BLOCK_X, (LY + BLOCK_Y - 1) / BLOCK_Y, (LZ + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    if (continue_step == 0) {
        init_arrays<<<grid, block>>>(d_f, d_g, d_rho, d_ux, d_uy, d_uz,
                                   d_phi, d_force_realx, d_force_realy, d_force_realz);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    } else {
        char dirname[256];
        snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

        char data_filename[512];
        snprintf(data_filename, sizeof(data_filename), "%s/%09d.dat", dirname, continue_step);

        FILE *f_data = fopen(data_filename, "rb");
        if (f_data == NULL) {
            fprintf(stderr, "Cannot open file: %s\n", data_filename);
            exit(EXIT_FAILURE);
        }

        size_t read_elements;
        read_elements = fread(h_rho, sizeof(double), LXYZ, f_data);
        if (read_elements != LXYZ) {
            fprintf(stderr, "Error reading h_rho data: read %zu of %d elements\n", 
                    read_elements, LXYZ);
            fclose(f_data);
            exit(EXIT_FAILURE);
        }

        read_elements = fread(h_ux, sizeof(double), LXYZ, f_data);
        if (read_elements != LXYZ) {
            fprintf(stderr, "Error reading h_ux data: read %zu of %d elements\n", 
                    read_elements, LXYZ);
            fclose(f_data);
            exit(EXIT_FAILURE);
        }

        read_elements = fread(h_uy, sizeof(double), LXYZ, f_data);
        if (read_elements != LXYZ) {
            fprintf(stderr, "Error reading h_uy data: read %zu of %d elements\n", 
                    read_elements, LXYZ);
            fclose(f_data);
            exit(EXIT_FAILURE);
        }

        read_elements = fread(h_uz, sizeof(double), LXYZ, f_data);
        if (read_elements != LXYZ) {
            fprintf(stderr, "Error reading h_uz data: read %zu of %d elements\n", 
                    read_elements, LXYZ);
            fclose(f_data);
            exit(EXIT_FAILURE);
        }

        read_elements = fread(h_phi, sizeof(double), LXYZ, f_data);
        if (read_elements != LXYZ) {
            fprintf(stderr, "Error reading h_phi data: read %zu of %d elements\n", 
                    read_elements, LXYZ);
            fclose(f_data);
            exit(EXIT_FAILURE);
        }

        fclose(f_data);

        CHECK_CUDA_ERROR(cudaMemcpy(d_rho, h_rho, LXYZ * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_ux,  h_ux,  LXYZ * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_uy,  h_uy,  LXYZ * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_uz,  h_uz,  LXYZ * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_phi, h_phi, LXYZ * sizeof(double), cudaMemcpyHostToDevice));

        char fg_filename[512];
        snprintf(fg_filename, sizeof(fg_filename), "%s/fg%09d.dat", dirname, continue_step);

        FILE *f_fg = fopen(fg_filename, "rb");
        if (f_fg == NULL) {
            fprintf(stderr, "Cannot open file: %s\n", fg_filename);
            exit(EXIT_FAILURE);
        }

        read_elements = fread(h_f, sizeof(double), NPOP * LXYZ, f_fg);
        if (read_elements != NPOP * LXYZ) {
            fprintf(stderr, "Error reading h_f data: read %zu of %d elements\n", 
                    read_elements, NPOP * LXYZ);
            fclose(f_fg);
            exit(EXIT_FAILURE);
        }

        read_elements = fread(h_g, sizeof(double), NPOP * LXYZ, f_fg);
        if (read_elements != NPOP * LXYZ) {
            fprintf(stderr, "Error reading h_g data: read %zu of %d elements\n", 
                    read_elements, NPOP * LXYZ);
            fclose(f_fg);
            exit(EXIT_FAILURE);
        }

        fclose(f_fg);

        CHECK_CUDA_ERROR(cudaMemcpy(d_f, h_f, NPOP * LXYZ * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_g, h_g, NPOP * LXYZ * sizeof(double), cudaMemcpyHostToDevice));

        forcing<<<grid, block>>>(d_force_realx, d_force_realy, d_force_realz, d_phi);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}

__global__ void init_arrays(double *f, double *g, double *rho, double *ux, double *uy, double *uz, 
                           double *phi, double *force_realx, double *force_realy, double *force_realz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LY + iy) * LX + ix;

    // Initialize macroscopic variables
    rho[idx] = 0.0;
    ux[idx]  = 0.0;
    uy[idx]  = 0.0;
    uz[idx]  = 0.0;
    
    // Initialize temperature field (phi)
    phi[idx] = 0.5;

    // Initialize force field
    force_realx[idx] = 0.0;
    force_realy[idx] = d_grav0 * d_beta * (phi[idx] - d_t0);
    force_realz[idx] = 0.0;

    // Initialize distribution functions
    double usq = 0.0;  // u^2 + v^2 + w^2 = 0 initially

    #pragma unroll
    for (int ip = 0; ip < NPOP; ip++) {
        double feq = d_tp[ip] * rho[idx] * (1.0 - 1.5 * usq);
        f[ip * LXYZ + idx] = feq - 0.5 * force_realy[idx] * d_ciy[ip] / (1.0/3.0) * feq;
        
        double geq = d_tp[ip] * phi[idx] * (1.0 - 1.5 * usq);
        g[ip * LXYZ + idx] = geq;
    }
}