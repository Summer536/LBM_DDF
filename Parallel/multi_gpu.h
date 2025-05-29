#ifndef MULTI_GPU_H
#define MULTI_GPU_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include "parameters.h"

typedef struct {
    int gpu_id;
    int y_start_global, y_end_global;    // 全局Y坐标范围
    int LY_local;                        // 本地Y方向网格数(不含halo)
    int LY_local_with_halo;             // 本地Y方向网格数(含halo)
    int local_size;                      // 本地总网格数: LX * LY_local_with_halo * LZ
    
    int neighbor_up, neighbor_down;      // Y+和Y-方向的邻居GPU ID
    bool is_top_boundary, is_bottom_boundary;  // 是否为物理边界GPU
    
    double *d_f_local, *d_g_local;
    double *d_rho_local, *d_ux_local, *d_uy_local, *d_uz_local, *d_phi_local;
    double *d_f_temp_local, *d_g_temp_local;
    double *d_force_realx_local, *d_force_realy_local, *d_force_realz_local;
    
    double *d_send_buffer_up, *d_send_buffer_down;
    double *d_recv_buffer_up, *d_recv_buffer_down;
    int buffer_size;                     // 每个缓冲区大小: LX * LZ * NPOP
    
    // CUDA streams
    cudaStream_t stream_compute;
    
    double *h_rho_local, *h_ux_local, *h_uy_local, *h_uz_local, *h_phi_local;
} GPU_Domain;

typedef struct {
    int num_gpus;                        // GPU数量
    GPU_Domain **domains;               // 所有GPU的域信息
    
    bool **p2p_accessible;
} Multi_GPU_Manager;

extern Multi_GPU_Manager *gpu_manager;

int multi_gpu_init(int num_gpus);
void multi_gpu_finalize();
void setup_gpu_domains();
void allocate_local_memory(GPU_Domain *domain);
void free_local_memory(GPU_Domain *domain);

void check_p2p_capability();
void enable_p2p_access();

void exchange_halo_f(GPU_Domain *domain);
void exchange_halo_g(GPU_Domain *domain);
void pack_send_data_f(GPU_Domain *domain, bool send_up);
void pack_send_data_g(GPU_Domain *domain, bool send_up);
void unpack_recv_data_f(GPU_Domain *domain, bool from_up);
void unpack_recv_data_g(GPU_Domain *domain, bool from_up);

__global__ void pack_f_data_kernel(double *f_local, double *send_buffer, int LY_local_with_halo, int layer_index, bool is_upper_layer);
__global__ void unpack_f_data_kernel(double *f_local, double *recv_buffer, int LY_local_with_halo, int layer_index, bool is_from_upper);
__global__ void pack_g_data_kernel(double *g_local, double *send_buffer, int LY_local_with_halo, int layer_index, bool is_upper_layer);
__global__ void unpack_g_data_kernel(double *g_local, double *recv_buffer, int LY_local_with_halo, int layer_index, bool is_from_upper);

void streaming_step_parallel(int num_gpus);
void collision_step_parallel(int num_gpus);
void macrovar_step_parallel(int num_gpus);

__device__ __host__ int get_local_index(int ix, int iy_local, int iz, int LY_local_with_halo);
__device__ __host__ int get_global_y_from_local(int iy_local, int y_start_global);
__device__ __host__ bool is_boundary_y_local(int iy_local, int LY_local_with_halo, bool is_top_gpu, bool is_bottom_gpu);

// 输出辅助函数声明
void write_gpu_data_to_file(FILE *f_data, GPU_Domain *domain, const char *var_name);

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#endif // MULTI_GPU_H 