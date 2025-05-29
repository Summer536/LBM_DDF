#include "multi_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Multi_GPU_Manager *gpu_manager = NULL;

__device__ __host__ int get_local_index(int ix, int iy_local, int iz, int LY_local_with_halo) {
    return (iz * LY_local_with_halo + iy_local) * LX + ix;
}

__device__ __host__ int get_global_y_from_local(int iy_local, int y_start_global) {
    return y_start_global + iy_local - HALO_WIDTH;
}

__device__ __host__ bool is_boundary_y_local(int iy_local, int LY_local_with_halo, bool is_top_gpu, bool is_bottom_gpu) {
    if (is_bottom_gpu && iy_local == HALO_WIDTH - 1) return true;
    if (is_top_gpu && iy_local == LY_local_with_halo - HALO_WIDTH) return true;
    return false;
}

int multi_gpu_init(int num_gpus) {
    int device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    
    if (num_gpus > device_count) {
        fprintf(stderr, "Error: Requested %d GPUs, but only %d available\n", num_gpus, device_count);
        return -1;
    }
    
    if (num_gpus > MAX_GPUS) {
        fprintf(stderr, "Error: Number of GPUs (%d) exceeds MAX_GPUS (%d)\n", num_gpus, MAX_GPUS);
        return -1;
    }
    
    if (LY % num_gpus != 0) {
        fprintf(stderr, "Error: LY (%d) must be divisible by number of GPUs (%d)\n", LY, num_gpus);
        return -1;
    }
    
    gpu_manager = (Multi_GPU_Manager*)malloc(sizeof(Multi_GPU_Manager));
    if (!gpu_manager) {
        fprintf(stderr, "Failed to allocate memory for gpu_manager\n");
        return -1;
    }
    
    gpu_manager->num_gpus = num_gpus;
    gpu_manager->domains = (GPU_Domain**)malloc(num_gpus * sizeof(GPU_Domain*));
    
    printf("Multi-GPU initialization successful:\n");
    printf("  Number of GPUs: %d\n", num_gpus);
    printf("  Global grid: %d x %d x %d\n", LX, LY, LZ);
    printf("  Local Y per GPU: %d\n", GET_LY_LOCAL(num_gpus));
    
    return 0;
}

void setup_gpu_domains() {
    for (int gpu_id = 0; gpu_id < gpu_manager->num_gpus; gpu_id++) {
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        
        GPU_Domain *domain = (GPU_Domain*)malloc(sizeof(GPU_Domain));
        gpu_manager->domains[gpu_id] = domain;
        
        domain->gpu_id = gpu_id;
        domain->LY_local = GET_LY_LOCAL(gpu_manager->num_gpus);
        domain->LY_local_with_halo = GET_LY_LOCAL_WITH_HALO(gpu_manager->num_gpus);
        domain->local_size = GET_LOCAL_SIZE(domain->LY_local_with_halo);
        domain->buffer_size = GET_BUFFER_SIZE();
        
        domain->y_start_global = gpu_id * domain->LY_local;
        domain->y_end_global = domain->y_start_global + domain->LY_local - 1;
        
        domain->neighbor_down = (gpu_id > 0) ? gpu_id - 1 : -1;
        domain->neighbor_up = (gpu_id < gpu_manager->num_gpus - 1) ? gpu_id + 1 : -1;
        
        domain->is_bottom_boundary = (gpu_id == 0);
        domain->is_top_boundary = (gpu_id == gpu_manager->num_gpus - 1);
        
        printf("GPU %d: Y range [%d, %d], local Y: %d (+%d halo), neighbors: down=%d, up=%d\n",
               gpu_id, domain->y_start_global, domain->y_end_global,
               domain->LY_local, 2 * HALO_WIDTH, domain->neighbor_down, domain->neighbor_up);
    }
}

void allocate_local_memory(GPU_Domain *domain) {
    CHECK_CUDA_ERROR(cudaSetDevice(domain->gpu_id));
    
    size_t field_size = domain->local_size * sizeof(double);
    size_t dist_size = NPOP * domain->local_size * sizeof(double);
    size_t buffer_size = domain->buffer_size * sizeof(double);

    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_f_local, dist_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_g_local, dist_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_f_temp_local, dist_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_g_temp_local, dist_size));
    
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_rho_local, field_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_ux_local, field_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_uy_local, field_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_uz_local, field_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_phi_local, field_size));
    
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_force_realx_local, field_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_force_realy_local, field_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_force_realz_local, field_size));
    
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_send_buffer_up, buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_send_buffer_down, buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_recv_buffer_up, buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&domain->d_recv_buffer_down, buffer_size));
    
    domain->h_rho_local = (double*)malloc(field_size);
    domain->h_ux_local = (double*)malloc(field_size);
    domain->h_uy_local = (double*)malloc(field_size);
    domain->h_uz_local = (double*)malloc(field_size);
    domain->h_phi_local = (double*)malloc(field_size);
    
    CHECK_CUDA_ERROR(cudaStreamCreate(&domain->stream_compute));
    
    printf("GPU %d: Memory allocated successfully\n", domain->gpu_id);
}

void check_p2p_capability() {
    int num_gpus = gpu_manager->num_gpus;
    
    gpu_manager->p2p_accessible = (bool**)malloc(num_gpus * sizeof(bool*));
    for (int i = 0; i < num_gpus; i++) {
        gpu_manager->p2p_accessible[i] = (bool*)malloc(num_gpus * sizeof(bool));
    }
    
    int can_access;
    for (int i = 0; i < num_gpus; i++) {
        for (int j = 0; j < num_gpus; j++) {
            if (i == j) {
                gpu_manager->p2p_accessible[i][j] = true;
            } else {
                CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&can_access, i, j));
                gpu_manager->p2p_accessible[i][j] = (can_access == 1);
            }
        }
    }
    
    printf("P2P accessibility matrix:\n");
    for (int i = 0; i < num_gpus; i++) {
        printf("GPU %d: ", i);
        for (int j = 0; j < num_gpus; j++) {
            printf("%d ", gpu_manager->p2p_accessible[i][j] ? 1 : 0);
        }
        printf("\n");
    }
}

void enable_p2p_access() {
    for (int i = 0; i < gpu_manager->num_gpus; i++) {
        GPU_Domain *domain = gpu_manager->domains[i];
        CHECK_CUDA_ERROR(cudaSetDevice(i));

        if (domain->neighbor_up >= 0 && 
            gpu_manager->p2p_accessible[i][domain->neighbor_up]) {
            CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(domain->neighbor_up, 0));
            printf("GPU %d: Enabled P2P access to GPU %d\n", i, domain->neighbor_up);
        }
        
        if (domain->neighbor_down >= 0 && 
            gpu_manager->p2p_accessible[i][domain->neighbor_down]) {
            CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(domain->neighbor_down, 0));
            printf("GPU %d: Enabled P2P access to GPU %d\n", i, domain->neighbor_down);
        }
    }
}

void exchange_halo_f(GPU_Domain *domain) {
    CHECK_CUDA_ERROR(cudaSetDevice(domain->gpu_id));
    
    dim3 block_size(16, 16);
    dim3 grid_size((LX + block_size.x - 1) / block_size.x,
                   (LZ + block_size.y - 1) / block_size.y);
    
    if (domain->neighbor_up >= 0) {
        pack_f_data_kernel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_f_local, domain->d_send_buffer_up, 
            domain->LY_local_with_halo, 0, true);
        
        CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(
            gpu_manager->domains[domain->neighbor_up]->d_recv_buffer_down,
            domain->neighbor_up,
            domain->d_send_buffer_up,
            domain->gpu_id,
            domain->buffer_size * sizeof(double),
            domain->stream_compute));
    }
    
    if (domain->neighbor_down >= 0) {
        pack_f_data_kernel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_f_local, domain->d_send_buffer_down, 
            domain->LY_local_with_halo, 0, false);
        
        CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(
            gpu_manager->domains[domain->neighbor_down]->d_recv_buffer_up,
            domain->neighbor_down,
            domain->d_send_buffer_down,
            domain->gpu_id,
            domain->buffer_size * sizeof(double),
            domain->stream_compute));
    }
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(domain->stream_compute));
    
    if (domain->neighbor_up >= 0) {
        unpack_f_data_kernel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_f_local, domain->d_recv_buffer_up,
            domain->LY_local_with_halo, 0, true);
    }
    
    if (domain->neighbor_down >= 0) {
        unpack_f_data_kernel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_f_local, domain->d_recv_buffer_down,
            domain->LY_local_with_halo, 0, false);
    }
}

void exchange_halo_g(GPU_Domain *domain) {
    CHECK_CUDA_ERROR(cudaSetDevice(domain->gpu_id));
    
    dim3 block_size(16, 16);
    dim3 grid_size((LX + block_size.x - 1) / block_size.x,
                   (LZ + block_size.y - 1) / block_size.y);
    
    if (domain->neighbor_up >= 0) {
        pack_g_data_kernel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_g_local, domain->d_send_buffer_up, 
            domain->LY_local_with_halo, 0, true);
        
        CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(
            gpu_manager->domains[domain->neighbor_up]->d_recv_buffer_down,
            domain->neighbor_up,
            domain->d_send_buffer_up,
            domain->gpu_id,
            domain->buffer_size * sizeof(double),
            domain->stream_compute));
    }
    
    if (domain->neighbor_down >= 0) {
        pack_g_data_kernel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_g_local, domain->d_send_buffer_down, 
            domain->LY_local_with_halo, 0, false);
        
        CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(
            gpu_manager->domains[domain->neighbor_down]->d_recv_buffer_up,
            domain->neighbor_down,
            domain->d_send_buffer_down,
            domain->gpu_id,
            domain->buffer_size * sizeof(double),
            domain->stream_compute));
    }
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(domain->stream_compute));
    
    if (domain->neighbor_up >= 0) {
        unpack_g_data_kernel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_g_local, domain->d_recv_buffer_up,
            domain->LY_local_with_halo, 0, true);
    }
    
    if (domain->neighbor_down >= 0) {
        unpack_g_data_kernel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_g_local, domain->d_recv_buffer_down,
            domain->LY_local_with_halo, 0, false);
    }
}

void pack_send_data_f(GPU_Domain *domain, bool send_up) {
    dim3 block_size(16, 16);
    dim3 grid_size((LX + block_size.x - 1) / block_size.x,
                   (LZ + block_size.y - 1) / block_size.y);
    
    double *send_buffer = send_up ? domain->d_send_buffer_up : domain->d_send_buffer_down;
    pack_f_data_kernel<<<grid_size, block_size>>>(
        domain->d_f_local, send_buffer, 
        domain->LY_local_with_halo, 0, send_up);
}

void pack_send_data_g(GPU_Domain *domain, bool send_up) {
    dim3 block_size(16, 16);
    dim3 grid_size((LX + block_size.x - 1) / block_size.x,
                   (LZ + block_size.y - 1) / block_size.y);
    
    double *send_buffer = send_up ? domain->d_send_buffer_up : domain->d_send_buffer_down;
    pack_g_data_kernel<<<grid_size, block_size>>>(
        domain->d_g_local, send_buffer, 
        domain->LY_local_with_halo, 0, send_up);
}

void unpack_recv_data_f(GPU_Domain *domain, bool from_up) {
    dim3 block_size(16, 16);
    dim3 grid_size((LX + block_size.x - 1) / block_size.x,
                   (LZ + block_size.y - 1) / block_size.y);
    
    double *recv_buffer = from_up ? domain->d_recv_buffer_up : domain->d_recv_buffer_down;
    unpack_f_data_kernel<<<grid_size, block_size>>>(
        domain->d_f_local, recv_buffer,
        domain->LY_local_with_halo, 0, from_up);
}

void unpack_recv_data_g(GPU_Domain *domain, bool from_up) {
    dim3 block_size(16, 16);
    dim3 grid_size((LX + block_size.x - 1) / block_size.x,
                   (LZ + block_size.y - 1) / block_size.y);
    
    double *recv_buffer = from_up ? domain->d_recv_buffer_up : domain->d_recv_buffer_down;
    unpack_g_data_kernel<<<grid_size, block_size>>>(
        domain->d_g_local, recv_buffer,
        domain->LY_local_with_halo, 0, from_up);
}

void free_local_memory(GPU_Domain *domain) {
    if (!domain) return;
    
    CHECK_CUDA_ERROR(cudaSetDevice(domain->gpu_id));
    
    cudaFree(domain->d_f_local);
    cudaFree(domain->d_g_local);
    cudaFree(domain->d_f_temp_local);
    cudaFree(domain->d_g_temp_local);
    cudaFree(domain->d_rho_local);
    cudaFree(domain->d_ux_local);
    cudaFree(domain->d_uy_local);
    cudaFree(domain->d_uz_local);
    cudaFree(domain->d_phi_local);
    cudaFree(domain->d_force_realx_local);
    cudaFree(domain->d_force_realy_local);
    cudaFree(domain->d_force_realz_local);
    cudaFree(domain->d_send_buffer_up);
    cudaFree(domain->d_send_buffer_down);
    cudaFree(domain->d_recv_buffer_up);
    cudaFree(domain->d_recv_buffer_down);
    
    free(domain->h_rho_local);
    free(domain->h_ux_local);
    free(domain->h_uy_local);
    free(domain->h_uz_local);
    free(domain->h_phi_local);
    
    cudaStreamDestroy(domain->stream_compute);
}

void multi_gpu_finalize() {
    if (gpu_manager) {
        if (gpu_manager->domains) {
            for (int i = 0; i < gpu_manager->num_gpus; i++) {
                if (gpu_manager->domains[i]) {
                    free_local_memory(gpu_manager->domains[i]);
                    free(gpu_manager->domains[i]);
                }
            }
            free(gpu_manager->domains);
        }
        
        if (gpu_manager->p2p_accessible) {
            for (int i = 0; i < gpu_manager->num_gpus; i++) {
                free(gpu_manager->p2p_accessible[i]);
            }
            free(gpu_manager->p2p_accessible);
        }
        
        free(gpu_manager);
        gpu_manager = NULL;
    }
} 