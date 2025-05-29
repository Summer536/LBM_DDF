#include "rb3d.h"
#include "multi_gpu.h"

// 多GPU并行streaming kernel - 支持局部Y坐标和halo区域
__global__ void streaming_parallel(double *f_local, double *f_temp_local, 
                                 int LY_local_with_halo, int y_start_global,
                                 bool is_top_boundary, bool is_bottom_boundary) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_local = blockIdx.y * blockDim.y + threadIdx.y;  // 本地Y坐标 (含halo)
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ix >= LX || iy_local >= LY_local_with_halo || iz >= LZ) return;

    int local_size = LX * LY_local_with_halo * LZ;
    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    
    int idx_x_pos = idx_local + 1;
    int idx_x_neg = idx_local - 1;
    int idx_y_pos = idx_local + LX;
    int idx_y_neg = idx_local - LX;
    int idx_z_pos = idx_local + LX * LY_local_with_halo;
    int idx_z_neg = idx_local - LX * LY_local_with_halo;

    // Center point (0)
    f_temp_local[idx_local] = f_local[idx_local];
    
    // Face neighbors
    // X方向 (无跨GPU通信)
    if (ix > 0)     f_temp_local[local_size + idx_local] = f_local[local_size + idx_x_neg];  // x- (1)
    if (ix < LX-1)  f_temp_local[3 * local_size + idx_local] = f_local[3 * local_size + idx_x_pos];  // x+ (3)
    
    // Y方向 (可能涉及跨GPU通信，但halo已更新)
    if (iy_local > 0)     f_temp_local[2 * local_size + idx_local] = f_local[2 * local_size + idx_y_neg];  // y- (2)
    if (iy_local < LY_local_with_halo-1)  f_temp_local[4 * local_size + idx_local] = f_local[4 * local_size + idx_y_pos];  // y+ (4)
    
    // Z方向 (无跨GPU通信)
    if (iz > 0)     f_temp_local[5 * local_size + idx_local] = f_local[5 * local_size + idx_z_neg];  // z- (5)
    if (iz < LZ-1)  f_temp_local[6 * local_size + idx_local] = f_local[6 * local_size + idx_z_pos];  // z+ (6)

    // Edge neighbors (xy-plane)
    if (ix > 0    && iy_local > 0)      f_temp_local[7  * local_size + idx_local] = f_local[7  * local_size + idx_x_neg + idx_y_neg - idx_local];
    if (ix < LX-1 && iy_local > 0)      f_temp_local[8  * local_size + idx_local] = f_local[8  * local_size + idx_x_pos + idx_y_neg - idx_local];
    if (ix < LX-1 && iy_local < LY_local_with_halo-1)   f_temp_local[9  * local_size + idx_local] = f_local[9  * local_size + idx_x_pos + idx_y_pos - idx_local];
    if (ix > 0    && iy_local < LY_local_with_halo-1)   f_temp_local[10 * local_size + idx_local] = f_local[10 * local_size + idx_x_neg + idx_y_pos - idx_local];

    // Edge neighbors (xz-plane)
    if (ix > 0    && iz > 0)      f_temp_local[11 * local_size + idx_local] = f_local[11 * local_size + idx_x_neg + idx_z_neg - idx_local];
    if (ix < LX-1 && iz > 0)      f_temp_local[13 * local_size + idx_local] = f_local[13 * local_size + idx_x_pos + idx_z_neg - idx_local];
    if (ix < LX-1 && iz < LZ-1)   f_temp_local[17 * local_size + idx_local] = f_local[17 * local_size + idx_x_pos + idx_z_pos - idx_local];
    if (ix > 0    && iz < LZ-1)   f_temp_local[15 * local_size + idx_local] = f_local[15 * local_size + idx_x_neg + idx_z_pos - idx_local];

    // Edge neighbors (yz-plane)
    if (iy_local > 0    && iz > 0)      f_temp_local[12 * local_size + idx_local] = f_local[12 * local_size + idx_y_neg + idx_z_neg - idx_local];
    if (iy_local < LY_local_with_halo-1 && iz > 0)      f_temp_local[14 * local_size + idx_local] = f_local[14 * local_size + idx_y_pos + idx_z_neg - idx_local];
    if (iy_local < LY_local_with_halo-1 && iz < LZ-1)   f_temp_local[18 * local_size + idx_local] = f_local[18 * local_size + idx_y_pos + idx_z_pos - idx_local];
    if (iy_local > 0    && iz < LZ-1)   f_temp_local[16 * local_size + idx_local] = f_local[16 * local_size + idx_y_neg + idx_z_pos - idx_local];

    // Corner neighbors
    if (ix > 0    && iy_local > 0    && iz > 0)        f_temp_local[19 * local_size + idx_local] = f_local[19 * local_size + idx_x_neg + idx_y_neg + idx_z_neg - 2 * idx_local];
    if (ix < LX-1 && iy_local > 0    && iz > 0)        f_temp_local[20 * local_size + idx_local] = f_local[20 * local_size + idx_x_pos + idx_y_neg + idx_z_neg - 2 * idx_local];
    if (ix < LX-1 && iy_local < LY_local_with_halo-1 && iz > 0)        f_temp_local[21 * local_size + idx_local] = f_local[21 * local_size + idx_x_pos + idx_y_pos + idx_z_neg - 2 * idx_local];
    if (ix > 0    && iy_local < LY_local_with_halo-1 && iz > 0)        f_temp_local[22 * local_size + idx_local] = f_local[22 * local_size + idx_x_neg + idx_y_pos + idx_z_neg - 2 * idx_local];
    if (ix > 0    && iy_local > 0    && iz < LZ-1)     f_temp_local[23 * local_size + idx_local] = f_local[23 * local_size + idx_x_neg + idx_y_neg + idx_z_pos - 2 * idx_local];
    if (ix < LX-1 && iy_local > 0    && iz < LZ-1)     f_temp_local[24 * local_size + idx_local] = f_local[24 * local_size + idx_x_pos + idx_y_neg + idx_z_pos - 2 * idx_local];
    if (ix < LX-1 && iy_local < LY_local_with_halo-1 && iz < LZ-1)     f_temp_local[25 * local_size + idx_local] = f_local[25 * local_size + idx_x_pos + idx_y_pos + idx_z_pos - 2 * idx_local];
    if (ix > 0    && iy_local < LY_local_with_halo-1 && iz < LZ-1)     f_temp_local[26 * local_size + idx_local] = f_local[26 * local_size + idx_x_neg + idx_y_pos + idx_z_pos - 2 * idx_local];

    // 物理边界条件 (固壁边界，速度为0)
    // X方向边界 (各GPU独立处理)
    if (ix == 0) {  // Left wall
        f_temp_local[local_size + idx_local] = f_local[3  * local_size + idx_local];
        f_temp_local[7  * local_size + idx_local] = f_local[9  * local_size + idx_local];
        f_temp_local[10 * local_size + idx_local] = f_local[8  * local_size + idx_local];
        f_temp_local[11 * local_size + idx_local] = f_local[17 * local_size + idx_local];
        f_temp_local[19 * local_size + idx_local] = f_local[25 * local_size + idx_local];
        f_temp_local[22 * local_size + idx_local] = f_local[24 * local_size + idx_local];
        f_temp_local[15 * local_size + idx_local] = f_local[13 * local_size + idx_local];
        f_temp_local[23 * local_size + idx_local] = f_local[21 * local_size + idx_local];
        f_temp_local[26 * local_size + idx_local] = f_local[20 * local_size + idx_local];
    }
    if (ix == LX-1) {  // Right wall
        f_temp_local[3  * local_size + idx_local] = f_local[local_size + idx_local];
        f_temp_local[9  * local_size + idx_local] = f_local[7  * local_size + idx_local];
        f_temp_local[8  * local_size + idx_local] = f_local[10 * local_size + idx_local];
        f_temp_local[17 * local_size + idx_local] = f_local[11 * local_size + idx_local];
        f_temp_local[25 * local_size + idx_local] = f_local[19 * local_size + idx_local];
        f_temp_local[24 * local_size + idx_local] = f_local[22 * local_size + idx_local];
        f_temp_local[13 * local_size + idx_local] = f_local[15 * local_size + idx_local];
        f_temp_local[21 * local_size + idx_local] = f_local[23 * local_size + idx_local];
        f_temp_local[20 * local_size + idx_local] = f_local[26 * local_size + idx_local];
    }
    
    // Y方向物理边界 (仅最上层和最下层GPU处理)
    int iy_global = get_global_y_from_local(iy_local, y_start_global);
    if (is_bottom_boundary && iy_global == 0) {  // Bottom wall (冷边界)
        f_temp_local[2  * local_size + idx_local] = f_local[4  * local_size + idx_local];
        f_temp_local[7  * local_size + idx_local] = f_local[9  * local_size + idx_local];
        f_temp_local[8  * local_size + idx_local] = f_local[10 * local_size + idx_local];
        f_temp_local[12 * local_size + idx_local] = f_local[18 * local_size + idx_local];
        f_temp_local[19 * local_size + idx_local] = f_local[25 * local_size + idx_local];
        f_temp_local[20 * local_size + idx_local] = f_local[26 * local_size + idx_local];
        f_temp_local[16 * local_size + idx_local] = f_local[14 * local_size + idx_local];
        f_temp_local[23 * local_size + idx_local] = f_local[21 * local_size + idx_local];
        f_temp_local[24 * local_size + idx_local] = f_local[22 * local_size + idx_local];
    }
    if (is_top_boundary && iy_global == LY-1) {  // Top wall (热边界)
        f_temp_local[4  * local_size + idx_local] = f_local[2  * local_size + idx_local];
        f_temp_local[9  * local_size + idx_local] = f_local[7  * local_size + idx_local];
        f_temp_local[10 * local_size + idx_local] = f_local[8  * local_size + idx_local];
        f_temp_local[18 * local_size + idx_local] = f_local[12 * local_size + idx_local];
        f_temp_local[25 * local_size + idx_local] = f_local[19 * local_size + idx_local];
        f_temp_local[26 * local_size + idx_local] = f_local[20 * local_size + idx_local];
        f_temp_local[14 * local_size + idx_local] = f_local[16 * local_size + idx_local];
        f_temp_local[21 * local_size + idx_local] = f_local[23 * local_size + idx_local];
        f_temp_local[22 * local_size + idx_local] = f_local[24 * local_size + idx_local];
    }
    
    // Z方向边界 (各GPU独立处理)
    if (iz == 0) {  // Front wall
        f_temp_local[5  * local_size + idx_local] = f_local[6  * local_size + idx_local];
        f_temp_local[11 * local_size + idx_local] = f_local[17 * local_size + idx_local];
        f_temp_local[12 * local_size + idx_local] = f_local[18 * local_size + idx_local];
        f_temp_local[13 * local_size + idx_local] = f_local[15 * local_size + idx_local];
        f_temp_local[14 * local_size + idx_local] = f_local[16 * local_size + idx_local];
        f_temp_local[19 * local_size + idx_local] = f_local[25 * local_size + idx_local];
        f_temp_local[20 * local_size + idx_local] = f_local[26 * local_size + idx_local];
        f_temp_local[21 * local_size + idx_local] = f_local[23 * local_size + idx_local];
        f_temp_local[22 * local_size + idx_local] = f_local[24 * local_size + idx_local];
    }
    if (iz == LZ-1) {  // Back wall
        f_temp_local[6  * local_size + idx_local] = f_local[5  * local_size + idx_local];
        f_temp_local[17 * local_size + idx_local] = f_local[11 * local_size + idx_local];
        f_temp_local[18 * local_size + idx_local] = f_local[12 * local_size + idx_local];
        f_temp_local[15 * local_size + idx_local] = f_local[13 * local_size + idx_local];
        f_temp_local[16 * local_size + idx_local] = f_local[14 * local_size + idx_local];
        f_temp_local[25 * local_size + idx_local] = f_local[19 * local_size + idx_local];
        f_temp_local[26 * local_size + idx_local] = f_local[20 * local_size + idx_local];
        f_temp_local[23 * local_size + idx_local] = f_local[21 * local_size + idx_local];
        f_temp_local[24 * local_size + idx_local] = f_local[22 * local_size + idx_local];
    }
}

// 多GPU并行streaming_scalar kernel
__global__ void streaming_scalar_parallel(double *g_local, double *g_temp_local, double *rho_local,
                                        int LY_local_with_halo, int y_start_global,
                                        bool is_top_boundary, bool is_bottom_boundary) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_local = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ix >= LX || iy_local >= LY_local_with_halo || iz >= LZ) return;

    int local_size = LX * LY_local_with_halo * LZ;
    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    
    // 计算邻居节点的本地索引
    int idx_x_pos = idx_local + 1;
    int idx_x_neg = idx_local - 1;
    int idx_y_pos = idx_local + LX;
    int idx_y_neg = idx_local - LX;
    int idx_z_pos = idx_local + LX * LY_local_with_halo;
    int idx_z_neg = idx_local - LX * LY_local_with_halo;

    // Center point
    g_temp_local[idx_local] = g_local[idx_local];
    
    // Face neighbors
    if (ix > 0)     g_temp_local[local_size + idx_local] = g_local[local_size + idx_x_neg];
    if (ix < LX-1)  g_temp_local[3 * local_size + idx_local] = g_local[3 * local_size + idx_x_pos];
    if (iy_local > 0)     g_temp_local[2 * local_size + idx_local] = g_local[2 * local_size + idx_y_neg];
    if (iy_local < LY_local_with_halo-1)  g_temp_local[4 * local_size + idx_local] = g_local[4 * local_size + idx_y_pos];
    if (iz > 0)     g_temp_local[5 * local_size + idx_local] = g_local[5 * local_size + idx_z_neg];
    if (iz < LZ-1)  g_temp_local[6 * local_size + idx_local] = g_local[6 * local_size + idx_z_pos];

    // Edge neighbors (xy-plane)
    if (ix > 0    && iy_local > 0)      g_temp_local[7  * local_size + idx_local] = g_local[7  * local_size + idx_x_neg + idx_y_neg - idx_local];
    if (ix < LX-1 && iy_local > 0)      g_temp_local[8  * local_size + idx_local] = g_local[8  * local_size + idx_x_pos + idx_y_neg - idx_local];
    if (ix < LX-1 && iy_local < LY_local_with_halo-1)   g_temp_local[9  * local_size + idx_local] = g_local[9  * local_size + idx_x_pos + idx_y_pos - idx_local];
    if (ix > 0    && iy_local < LY_local_with_halo-1)   g_temp_local[10 * local_size + idx_local] = g_local[10 * local_size + idx_x_neg + idx_y_pos - idx_local];

    // Edge neighbors (xz-plane)
    if (ix > 0    && iz > 0)      g_temp_local[11 * local_size + idx_local] = g_local[11 * local_size + idx_x_neg + idx_z_neg - idx_local];
    if (ix < LX-1 && iz > 0)      g_temp_local[13 * local_size + idx_local] = g_local[13 * local_size + idx_x_pos + idx_z_neg - idx_local];
    if (ix < LX-1 && iz < LZ-1)   g_temp_local[17 * local_size + idx_local] = g_local[17 * local_size + idx_x_pos + idx_z_pos - idx_local];
    if (ix > 0    && iz < LZ-1)   g_temp_local[15 * local_size + idx_local] = g_local[15 * local_size + idx_x_neg + idx_z_pos - idx_local];

    // Edge neighbors (yz-plane)
    if (iy_local > 0    && iz > 0)      g_temp_local[12 * local_size + idx_local] = g_local[12 * local_size + idx_y_neg + idx_z_neg - idx_local];
    if (iy_local < LY_local_with_halo-1 && iz > 0)      g_temp_local[14 * local_size + idx_local] = g_local[14 * local_size + idx_y_pos + idx_z_neg - idx_local];
    if (iy_local < LY_local_with_halo-1 && iz < LZ-1)   g_temp_local[18 * local_size + idx_local] = g_local[18 * local_size + idx_y_pos + idx_z_pos - idx_local];
    if (iy_local > 0    && iz < LZ-1)   g_temp_local[16 * local_size + idx_local] = g_local[16 * local_size + idx_y_neg + idx_z_pos - idx_local];

    // Corner neighbors
    if (ix > 0    && iy_local > 0    && iz > 0)        g_temp_local[19 * local_size + idx_local] = g_local[19 * local_size + idx_x_neg + idx_y_neg + idx_z_neg - 2 * idx_local];
    if (ix < LX-1 && iy_local > 0    && iz > 0)        g_temp_local[20 * local_size + idx_local] = g_local[20 * local_size + idx_x_pos + idx_y_neg + idx_z_neg - 2 * idx_local];
    if (ix < LX-1 && iy_local < LY_local_with_halo-1 && iz > 0)        g_temp_local[21 * local_size + idx_local] = g_local[21 * local_size + idx_x_pos + idx_y_pos + idx_z_neg - 2 * idx_local];
    if (ix > 0    && iy_local < LY_local_with_halo-1 && iz > 0)        g_temp_local[22 * local_size + idx_local] = g_local[22 * local_size + idx_x_neg + idx_y_pos + idx_z_neg - 2 * idx_local];
    if (ix > 0    && iy_local > 0    && iz < LZ-1)     g_temp_local[23 * local_size + idx_local] = g_local[23 * local_size + idx_x_neg + idx_y_neg + idx_z_pos - 2 * idx_local];
    if (ix < LX-1 && iy_local > 0    && iz < LZ-1)     g_temp_local[24 * local_size + idx_local] = g_local[24 * local_size + idx_x_pos + idx_y_neg + idx_z_pos - 2 * idx_local];
    if (ix < LX-1 && iy_local < LY_local_with_halo-1 && iz < LZ-1)     g_temp_local[25 * local_size + idx_local] = g_local[25 * local_size + idx_x_pos + idx_y_pos + idx_z_pos - 2 * idx_local];
    if (ix > 0    && iy_local < LY_local_with_halo-1 && iz < LZ-1)     g_temp_local[26 * local_size + idx_local] = g_local[26 * local_size + idx_x_neg + idx_y_pos + idx_z_pos - 2 * idx_local];

    // 温度边界条件
    int iy_global = get_global_y_from_local(iy_local, y_start_global);
    
    // X和Z方向绝热边界 (反射边界条件)
    if (ix == 0) {  // Left wall (adiabatic)
        g_temp_local[local_size + idx_local] = g_local[3  * local_size + idx_local];
        g_temp_local[7  * local_size + idx_local] = g_local[9  * local_size + idx_local];
        g_temp_local[10 * local_size + idx_local] = g_local[8  * local_size + idx_local];
        g_temp_local[11 * local_size + idx_local] = g_local[17 * local_size + idx_local];
        g_temp_local[19 * local_size + idx_local] = g_local[25 * local_size + idx_local];
        g_temp_local[22 * local_size + idx_local] = g_local[24 * local_size + idx_local];
        g_temp_local[15 * local_size + idx_local] = g_local[13 * local_size + idx_local];
        g_temp_local[23 * local_size + idx_local] = g_local[21 * local_size + idx_local];
        g_temp_local[26 * local_size + idx_local] = g_local[20 * local_size + idx_local];
    }
    if (ix == LX-1) {  // Right wall (adiabatic)
        g_temp_local[3  * local_size + idx_local] = g_local[local_size + idx_local];
        g_temp_local[9  * local_size + idx_local] = g_local[7  * local_size + idx_local];
        g_temp_local[8  * local_size + idx_local] = g_local[10 * local_size + idx_local];
        g_temp_local[17 * local_size + idx_local] = g_local[11 * local_size + idx_local];
        g_temp_local[25 * local_size + idx_local] = g_local[19 * local_size + idx_local];
        g_temp_local[24 * local_size + idx_local] = g_local[22 * local_size + idx_local];
        g_temp_local[13 * local_size + idx_local] = g_local[15 * local_size + idx_local];
        g_temp_local[21 * local_size + idx_local] = g_local[23 * local_size + idx_local];
        g_temp_local[20 * local_size + idx_local] = g_local[26 * local_size + idx_local];
    }
    
    // Y方向温度边界条件 (需要实际温度边界条件实现)
    if (is_bottom_boundary && iy_global == 0) {  // Bottom wall (T_COLD)
        double density = rho_local[idx_local];
        g_temp_local[2  * local_size + idx_local] = -g_local[4  * local_size + idx_local] + 2.0 * d_tp[2]  * d_tCold * (1.0 + density);
        g_temp_local[7  * local_size + idx_local] = -g_local[9  * local_size + idx_local] + 2.0 * d_tp[7]  * d_tCold * (1.0 + density);
        g_temp_local[8  * local_size + idx_local] = -g_local[10 * local_size + idx_local] + 2.0 * d_tp[8]  * d_tCold * (1.0 + density);
        g_temp_local[12 * local_size + idx_local] = -g_local[18 * local_size + idx_local] + 2.0 * d_tp[12] * d_tCold * (1.0 + density);
        g_temp_local[19 * local_size + idx_local] = -g_local[25 * local_size + idx_local] + 2.0 * d_tp[19] * d_tCold * (1.0 + density);
        g_temp_local[20 * local_size + idx_local] = -g_local[26 * local_size + idx_local] + 2.0 * d_tp[20] * d_tCold * (1.0 + density);
        g_temp_local[16 * local_size + idx_local] = -g_local[14 * local_size + idx_local] + 2.0 * d_tp[16] * d_tCold * (1.0 + density);
        g_temp_local[23 * local_size + idx_local] = -g_local[21 * local_size + idx_local] + 2.0 * d_tp[23] * d_tCold * (1.0 + density);
        g_temp_local[24 * local_size + idx_local] = -g_local[22 * local_size + idx_local] + 2.0 * d_tp[24] * d_tCold * (1.0 + density);
    }
    if (is_top_boundary && iy_global == LY-1) {  // Top wall (T_HOT)
        double density = rho_local[idx_local];
        g_temp_local[4  * local_size + idx_local] = -g_local[2  * local_size + idx_local] + 2.0 * d_tp[4]  * d_tHot * (1.0 + density);
        g_temp_local[9  * local_size + idx_local] = -g_local[7  * local_size + idx_local] + 2.0 * d_tp[9]  * d_tHot * (1.0 + density);
        g_temp_local[10 * local_size + idx_local] = -g_local[8  * local_size + idx_local] + 2.0 * d_tp[10] * d_tHot * (1.0 + density);
        g_temp_local[18 * local_size + idx_local] = -g_local[12 * local_size + idx_local] + 2.0 * d_tp[18] * d_tHot * (1.0 + density);
        g_temp_local[25 * local_size + idx_local] = -g_local[19 * local_size + idx_local] + 2.0 * d_tp[25] * d_tHot * (1.0 + density);
        g_temp_local[26 * local_size + idx_local] = -g_local[20 * local_size + idx_local] + 2.0 * d_tp[26] * d_tHot * (1.0 + density);
        g_temp_local[14 * local_size + idx_local] = -g_local[16 * local_size + idx_local] + 2.0 * d_tp[14] * d_tHot * (1.0 + density);
        g_temp_local[21 * local_size + idx_local] = -g_local[23 * local_size + idx_local] + 2.0 * d_tp[21] * d_tHot * (1.0 + density);
        g_temp_local[22 * local_size + idx_local] = -g_local[24 * local_size + idx_local] + 2.0 * d_tp[22] * d_tHot * (1.0 + density);
    }
    
    // Z方向绝热边界
    if (iz == 0) {  // Front wall (adiabatic)
        g_temp_local[5  * local_size + idx_local] = g_local[6  * local_size + idx_local];
        g_temp_local[11 * local_size + idx_local] = g_local[17 * local_size + idx_local];
        g_temp_local[12 * local_size + idx_local] = g_local[18 * local_size + idx_local];
        g_temp_local[13 * local_size + idx_local] = g_local[15 * local_size + idx_local];
        g_temp_local[14 * local_size + idx_local] = g_local[16 * local_size + idx_local];
        g_temp_local[19 * local_size + idx_local] = g_local[25 * local_size + idx_local];
        g_temp_local[20 * local_size + idx_local] = g_local[26 * local_size + idx_local];
        g_temp_local[21 * local_size + idx_local] = g_local[23 * local_size + idx_local];
        g_temp_local[22 * local_size + idx_local] = g_local[24 * local_size + idx_local];
    }
    if (iz == LZ-1) {  // Back wall (adiabatic)
        g_temp_local[6  * local_size + idx_local] = g_local[5  * local_size + idx_local];
        g_temp_local[17 * local_size + idx_local] = g_local[11 * local_size + idx_local];
        g_temp_local[18 * local_size + idx_local] = g_local[12 * local_size + idx_local];
        g_temp_local[15 * local_size + idx_local] = g_local[13 * local_size + idx_local];
        g_temp_local[16 * local_size + idx_local] = g_local[14 * local_size + idx_local];
        g_temp_local[25 * local_size + idx_local] = g_local[19 * local_size + idx_local];
        g_temp_local[26 * local_size + idx_local] = g_local[20 * local_size + idx_local];
        g_temp_local[23 * local_size + idx_local] = g_local[21 * local_size + idx_local];
        g_temp_local[24 * local_size + idx_local] = g_local[22 * local_size + idx_local];
    }
}

// =========== HALO数据交换实现 ===========

// 打包发送数据 - f分布函数
__global__ void pack_f_data_kernel(double *f_local, double *send_buffer, int LY_local_with_halo, int layer_index, bool is_upper_layer) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= LX || iz >= LZ) return;
    
    int iy_local = is_upper_layer ? (LY_local_with_halo - HALO_WIDTH - 1) : HALO_WIDTH;
    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    int buffer_idx = iz * LX + ix;
    int local_size = LX * LY_local_with_halo * LZ;
    
    // 打包所有速度方向的分布函数
    for (int pop = 0; pop < NPOP; pop++) {
        send_buffer[pop * LX * LZ + buffer_idx] = f_local[pop * local_size + idx_local];
    }
}

// 解包接收数据 - f分布函数
__global__ void unpack_f_data_kernel(double *f_local, double *recv_buffer, int LY_local_with_halo, int layer_index, bool is_from_upper) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= LX || iz >= LZ) return;
    
    int iy_local = is_from_upper ? (LY_local_with_halo - HALO_WIDTH + layer_index) : (layer_index);
    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    int buffer_idx = iz * LX + ix;
    int local_size = LX * LY_local_with_halo * LZ;
    
    for (int pop = 0; pop < NPOP; pop++) {
        f_local[pop * local_size + idx_local] = recv_buffer[pop * LX * LZ + buffer_idx];
    }
}

// 打包发送数据 - g分布函数
__global__ void pack_g_data_kernel(double *g_local, double *send_buffer, int LY_local_with_halo, int layer_index, bool is_upper_layer) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= LX || iz >= LZ) return;
    
    int iy_local = is_upper_layer ? (LY_local_with_halo - HALO_WIDTH - 1) : HALO_WIDTH;
    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    int buffer_idx = iz * LX + ix;
    int local_size = LX * LY_local_with_halo * LZ;
    
    for (int pop = 0; pop < NPOP; pop++) {
        send_buffer[pop * LX * LZ + buffer_idx] = g_local[pop * local_size + idx_local];
    }
}

// 解包接收数据 - g分布函数
__global__ void unpack_g_data_kernel(double *g_local, double *recv_buffer, int LY_local_with_halo, int layer_index, bool is_from_upper) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= LX || iz >= LZ) return;
    
    int iy_local = is_from_upper ? (LY_local_with_halo - HALO_WIDTH + layer_index) : (layer_index);
    int idx_local = get_local_index(ix, iy_local, iz, LY_local_with_halo);
    int buffer_idx = iz * LX + ix;
    int local_size = LX * LY_local_with_halo * LZ;
    
    for (int pop = 0; pop < NPOP; pop++) {
        g_local[pop * local_size + idx_local] = recv_buffer[pop * LX * LZ + buffer_idx];
    }
}

// 主GPU间并行流步函数
void streaming_step_parallel(int num_gpus) {
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        GPU_Domain *domain = gpu_manager->domains[gpu_id];
        exchange_halo_f(domain);
        exchange_halo_g(domain);
    }
    
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        GPU_Domain *domain = gpu_manager->domains[gpu_id];
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        
        dim3 block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);
        dim3 grid_size((LX + block_size.x - 1) / block_size.x,
                      (domain->LY_local_with_halo + block_size.y - 1) / block_size.y,
                      (LZ + block_size.z - 1) / block_size.z);
        
        streaming_parallel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_f_local, domain->d_f_temp_local, 
            domain->LY_local_with_halo, domain->y_start_global,
            domain->is_top_boundary, domain->is_bottom_boundary);
        
        streaming_scalar_parallel<<<grid_size, block_size, 0, domain->stream_compute>>>(
            domain->d_g_local, domain->d_g_temp_local, domain->d_rho_local,
            domain->LY_local_with_halo, domain->y_start_global,
            domain->is_top_boundary, domain->is_bottom_boundary);
            
        size_t dist_size = NPOP * domain->local_size * sizeof(double);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(domain->d_f_local, domain->d_f_temp_local, 
                                        dist_size, cudaMemcpyDeviceToDevice, domain->stream_compute));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(domain->d_g_local, domain->d_g_temp_local, 
                                        dist_size, cudaMemcpyDeviceToDevice, domain->stream_compute));
    }
}