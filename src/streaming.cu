#include "rb3d.h"

__global__ void streaming(double *f, double *f_temp) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LY + iy) * LX + ix;
    
    // Precompute offsets
    int idx_x_pos = idx + 1;
    int idx_x_neg = idx - 1;
    int idx_y_pos = idx + LX;
    int idx_y_neg = idx - LX;
    int idx_z_pos = idx + LX * LY;
    int idx_z_neg = idx - LX * LY;

    // Center point
    f_temp[idx] = f[idx];
    
    // Face neighbors
    if (ix > 0)     f_temp[    LXYZ + idx] = f[    LXYZ + idx_x_neg];  // x-
    if (ix < LX-1)  f_temp[3 * LXYZ + idx] = f[3 * LXYZ + idx_x_pos];  // x+
    if (iy > 0)     f_temp[2 * LXYZ + idx] = f[2 * LXYZ + idx_y_neg];  // y-
    if (iy < LY-1)  f_temp[4 * LXYZ + idx] = f[4 * LXYZ + idx_y_pos];  // y+
    if (iz > 0)     f_temp[5 * LXYZ + idx] = f[5 * LXYZ + idx_z_neg];  // z-
    if (iz < LZ-1)  f_temp[6 * LXYZ + idx] = f[6 * LXYZ + idx_z_pos];  // z+

    // Edge neighbors (xy-plane)
    if (ix > 0    && iy > 0)      f_temp[7  * LXYZ + idx] = f[7  * LXYZ + idx_x_neg + idx_y_neg - idx];
    if (ix < LX-1 && iy > 0)      f_temp[8  * LXYZ + idx] = f[8  * LXYZ + idx_x_pos + idx_y_neg - idx];
    if (ix < LX-1 && iy < LY-1)   f_temp[9  * LXYZ + idx] = f[9  * LXYZ + idx_x_pos + idx_y_pos - idx];
    if (ix > 0    && iy < LY-1)   f_temp[10 * LXYZ + idx] = f[10 * LXYZ + idx_x_neg + idx_y_pos - idx];

    // Edge neighbors (xz-plane)
    if (ix > 0    && iz > 0)      f_temp[11 * LXYZ + idx] = f[11 * LXYZ + idx_x_neg + idx_z_neg - idx];
    if (ix < LX-1 && iz > 0)      f_temp[13 * LXYZ + idx] = f[13 * LXYZ + idx_x_pos + idx_z_neg - idx];
    if (ix < LX-1 && iz < LZ-1)   f_temp[17 * LXYZ + idx] = f[17 * LXYZ + idx_x_pos + idx_z_pos - idx];
    if (ix > 0    && iz < LZ-1)   f_temp[15 * LXYZ + idx] = f[15 * LXYZ + idx_x_neg + idx_z_pos - idx];

    // Edge neighbors (yz-plane)
    if (iy > 0    && iz > 0)      f_temp[12 * LXYZ + idx] = f[12 * LXYZ + idx_y_neg + idx_z_neg - idx];
    if (iy < LY-1 && iz > 0)      f_temp[14 * LXYZ + idx] = f[14 * LXYZ + idx_y_pos + idx_z_neg - idx];
    if (iy < LY-1 && iz < LZ-1)   f_temp[18 * LXYZ + idx] = f[18 * LXYZ + idx_y_pos + idx_z_pos - idx];
    if (iy > 0    && iz < LZ-1)   f_temp[16 * LXYZ + idx] = f[16 * LXYZ + idx_y_neg + idx_z_pos - idx];

    // Corner neighbors
    if (ix > 0    && iy > 0    && iz > 0)        f_temp[19 * LXYZ + idx] = f[19 * LXYZ + idx_x_neg + idx_y_neg + idx_z_neg - 2 * idx];
    if (ix < LX-1 && iy > 0    && iz > 0)        f_temp[20 * LXYZ + idx] = f[20 * LXYZ + idx_x_pos + idx_y_neg + idx_z_neg - 2 * idx];
    if (ix < LX-1 && iy < LY-1 && iz > 0)        f_temp[21 * LXYZ + idx] = f[21 * LXYZ + idx_x_pos + idx_y_pos + idx_z_neg - 2 * idx];
    if (ix > 0    && iy < LY-1 && iz > 0)        f_temp[22 * LXYZ + idx] = f[22 * LXYZ + idx_x_neg + idx_y_pos + idx_z_neg - 2 * idx];
    if (ix > 0    && iy > 0    && iz < LZ-1)     f_temp[23 * LXYZ + idx] = f[23 * LXYZ + idx_x_neg + idx_y_neg + idx_z_pos - 2 * idx];
    if (ix < LX-1 && iy > 0    && iz < LZ-1)     f_temp[24 * LXYZ + idx] = f[24 * LXYZ + idx_x_pos + idx_y_neg + idx_z_pos - 2 * idx];
    if (ix < LX-1 && iy < LY-1 && iz < LZ-1)     f_temp[25 * LXYZ + idx] = f[25 * LXYZ + idx_x_pos + idx_y_pos + idx_z_pos - 2 * idx];
    if (ix > 0    && iy < LY-1 && iz < LZ-1)     f_temp[26 * LXYZ + idx] = f[26 * LXYZ + idx_x_neg + idx_y_pos + idx_z_pos - 2 * idx];

    // Boundary conditions
    if (ix == 0) {  // Left wall
        f_temp[     LXYZ + idx] = f[3  * LXYZ + idx];
        f_temp[7  * LXYZ + idx] = f[9  * LXYZ + idx];
        f_temp[10 * LXYZ + idx] = f[8  * LXYZ + idx];
        f_temp[11 * LXYZ + idx] = f[17 * LXYZ + idx];
        f_temp[19 * LXYZ + idx] = f[25 * LXYZ + idx];
        f_temp[22 * LXYZ + idx] = f[24 * LXYZ + idx];
        f_temp[15 * LXYZ + idx] = f[13 * LXYZ + idx];
        f_temp[23 * LXYZ + idx] = f[21 * LXYZ + idx];
        f_temp[26 * LXYZ + idx] = f[20 * LXYZ + idx];
    }
    if (ix == LX-1) {  // Right wall
        f_temp[3  * LXYZ + idx] = f[     LXYZ + idx];
        f_temp[9  * LXYZ + idx] = f[7  * LXYZ + idx];
        f_temp[8  * LXYZ + idx] = f[10 * LXYZ + idx];
        f_temp[17 * LXYZ + idx] = f[11 * LXYZ + idx];
        f_temp[25 * LXYZ + idx] = f[19 * LXYZ + idx];
        f_temp[24 * LXYZ + idx] = f[22 * LXYZ + idx];
        f_temp[13 * LXYZ + idx] = f[15 * LXYZ + idx];
        f_temp[21 * LXYZ + idx] = f[23 * LXYZ + idx];
        f_temp[20 * LXYZ + idx] = f[26 * LXYZ + idx];
    }
    if (iy == 0) {  // Bottom wall
        f_temp[2  * LXYZ + idx] = f[4  * LXYZ + idx];
        f_temp[7  * LXYZ + idx] = f[9  * LXYZ + idx];
        f_temp[8  * LXYZ + idx] = f[10 * LXYZ + idx];
        f_temp[12 * LXYZ + idx] = f[18 * LXYZ + idx];
        f_temp[19 * LXYZ + idx] = f[25 * LXYZ + idx];
        f_temp[20 * LXYZ + idx] = f[26 * LXYZ + idx];
        f_temp[16 * LXYZ + idx] = f[14 * LXYZ + idx];
        f_temp[23 * LXYZ + idx] = f[21 * LXYZ + idx];
        f_temp[24 * LXYZ + idx] = f[22 * LXYZ + idx];
    }
    if (iy == LY-1) {  // Top wall
        f_temp[4  * LXYZ + idx] = f[2  * LXYZ + idx];
        f_temp[9  * LXYZ + idx] = f[7  * LXYZ + idx];
        f_temp[10 * LXYZ + idx] = f[8  * LXYZ + idx];
        f_temp[18 * LXYZ + idx] = f[12 * LXYZ + idx];
        f_temp[25 * LXYZ + idx] = f[19 * LXYZ + idx];
        f_temp[26 * LXYZ + idx] = f[20 * LXYZ + idx];
        f_temp[14 * LXYZ + idx] = f[16 * LXYZ + idx];
        f_temp[21 * LXYZ + idx] = f[23 * LXYZ + idx];
        f_temp[22 * LXYZ + idx] = f[24 * LXYZ + idx];
    }
    if (iz == 0) {  // Front wall
        f_temp[5  * LXYZ + idx] = f[6  * LXYZ + idx];
        f_temp[11 * LXYZ + idx] = f[17 * LXYZ + idx];
        f_temp[12 * LXYZ + idx] = f[18 * LXYZ + idx];
        f_temp[13 * LXYZ + idx] = f[15 * LXYZ + idx];
        f_temp[14 * LXYZ + idx] = f[16 * LXYZ + idx];
        f_temp[19 * LXYZ + idx] = f[25 * LXYZ + idx];
        f_temp[20 * LXYZ + idx] = f[26 * LXYZ + idx];
        f_temp[21 * LXYZ + idx] = f[23 * LXYZ + idx];
        f_temp[22 * LXYZ + idx] = f[24 * LXYZ + idx];
    }
    if (iz == LZ-1) {  // Back wall
        f_temp[6  * LXYZ + idx] = f[5  * LXYZ + idx];
        f_temp[17 * LXYZ + idx] = f[11 * LXYZ + idx];
        f_temp[18 * LXYZ + idx] = f[12 * LXYZ + idx];
        f_temp[15 * LXYZ + idx] = f[13 * LXYZ + idx];
        f_temp[16 * LXYZ + idx] = f[14 * LXYZ + idx];
        f_temp[25 * LXYZ + idx] = f[19 * LXYZ + idx];
        f_temp[26 * LXYZ + idx] = f[20 * LXYZ + idx];
        f_temp[23 * LXYZ + idx] = f[21 * LXYZ + idx];
        f_temp[24 * LXYZ + idx] = f[22 * LXYZ + idx];
    }
}

__global__ void streaming_scalar(double *g, double *g_temp, double *rho) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= LX || iy >= LY || iz >= LZ) return;

    int idx = (iz * LY + iy) * LX + ix;

    // Precompute offsets
    int idx_x_pos = idx + 1;
    int idx_x_neg = idx - 1;
    int idx_y_pos = idx + LX;
    int idx_y_neg = idx - LX;
    int idx_z_pos = idx + LX * LY;
    int idx_z_neg = idx - LX * LY;

    // Center point
    g_temp[idx] = g[idx];
    
    // Face neighbors
    if (ix > 0)     g_temp[    LXYZ + idx] = g[    LXYZ + idx_x_neg];   // x-
    if (ix < LX-1)  g_temp[3 * LXYZ + idx] = g[3 * LXYZ + idx_x_pos];   // x+
    if (iy > 0)     g_temp[2 * LXYZ + idx] = g[2 * LXYZ + idx_y_neg];   // y-
    if (iy < LY-1)  g_temp[4 * LXYZ + idx] = g[4 * LXYZ + idx_y_pos];   // y+
    if (iz > 0)     g_temp[5 * LXYZ + idx] = g[5 * LXYZ + idx_z_neg];   // z-
    if (iz < LZ-1)  g_temp[6 * LXYZ + idx] = g[6 * LXYZ + idx_z_pos];   // z+

    // Edge neighbors (xy-plane)
    if (ix > 0    && iy > 0)      g_temp[7  * LXYZ + idx] = g[7  * LXYZ + idx_x_neg + idx_y_neg - idx];
    if (ix < LX-1 && iy > 0)      g_temp[8  * LXYZ + idx] = g[8  * LXYZ + idx_x_pos + idx_y_neg - idx];
    if (ix < LX-1 && iy < LY-1)   g_temp[9  * LXYZ + idx] = g[9  * LXYZ + idx_x_pos + idx_y_pos - idx];
    if (ix > 0    && iy < LY-1)   g_temp[10 * LXYZ + idx] = g[10 * LXYZ + idx_x_neg + idx_y_pos - idx];

    // Edge neighbors (xz-plane)
    if (ix > 0    && iz > 0)      g_temp[11 * LXYZ + idx] = g[11 * LXYZ + idx_x_neg + idx_z_neg - idx];
    if (ix < LX-1 && iz > 0)      g_temp[13 * LXYZ + idx] = g[13 * LXYZ + idx_x_pos + idx_z_neg - idx];
    if (ix < LX-1 && iz < LZ-1)   g_temp[17 * LXYZ + idx] = g[17 * LXYZ + idx_x_pos + idx_z_pos - idx];
    if (ix > 0    && iz < LZ-1)   g_temp[15 * LXYZ + idx] = g[15 * LXYZ + idx_x_neg + idx_z_pos - idx];

    // Edge neighbors (yz-plane)
    if (iy > 0    && iz > 0)      g_temp[12 * LXYZ + idx] = g[12 * LXYZ + idx_y_neg + idx_z_neg - idx];
    if (iy < LY-1 && iz > 0)      g_temp[14 * LXYZ + idx] = g[14 * LXYZ + idx_y_pos + idx_z_neg - idx];
    if (iy < LY-1 && iz < LZ-1)   g_temp[18 * LXYZ + idx] = g[18 * LXYZ + idx_y_pos + idx_z_pos - idx];
    if (iy > 0    && iz < LZ-1)   g_temp[16 * LXYZ + idx] = g[16 * LXYZ + idx_y_neg + idx_z_pos - idx];

    // Corner neighbors
    if (ix > 0    && iy > 0    && iz > 0)      g_temp[19 * LXYZ + idx] = g[19 * LXYZ + idx_x_neg + idx_y_neg + idx_z_neg - 2 * idx];
    if (ix < LX-1 && iy > 0    && iz > 0)      g_temp[20 * LXYZ + idx] = g[20 * LXYZ + idx_x_pos + idx_y_neg + idx_z_neg - 2 * idx];
    if (ix < LX-1 && iy < LY-1 && iz > 0)      g_temp[21 * LXYZ + idx] = g[21 * LXYZ + idx_x_pos + idx_y_pos + idx_z_neg - 2 * idx];
    if (ix > 0    && iy < LY-1 && iz > 0)      g_temp[22 * LXYZ + idx] = g[22 * LXYZ + idx_x_neg + idx_y_pos + idx_z_neg - 2 * idx];
    if (ix > 0    && iy > 0    && iz < LZ-1)   g_temp[23 * LXYZ + idx] = g[23 * LXYZ + idx_x_neg + idx_y_neg + idx_z_pos - 2 * idx];
    if (ix < LX-1 && iy > 0    && iz < LZ-1)   g_temp[24 * LXYZ + idx] = g[24 * LXYZ + idx_x_pos + idx_y_neg + idx_z_pos - 2 * idx];
    if (ix < LX-1 && iy < LY-1 && iz < LZ-1)   g_temp[25 * LXYZ + idx] = g[25 * LXYZ + idx_x_pos + idx_y_pos + idx_z_pos - 2 * idx];
    if (ix > 0    && iy < LY-1 && iz < LZ-1)   g_temp[26 * LXYZ + idx] = g[26 * LXYZ + idx_x_neg + idx_y_pos + idx_z_pos - 2 * idx];

    // Boundary conditions
    if (ix == 0) {  // Left wall (adiabatic)
        g_temp[     LXYZ + idx] = g[3  * LXYZ + idx];
        g_temp[7  * LXYZ + idx] = g[9  * LXYZ + idx];
        g_temp[10 * LXYZ + idx] = g[8  * LXYZ + idx];
        g_temp[11 * LXYZ + idx] = g[17 * LXYZ + idx];
        g_temp[19 * LXYZ + idx] = g[25 * LXYZ + idx];
        g_temp[22 * LXYZ + idx] = g[24 * LXYZ + idx];
        g_temp[15 * LXYZ + idx] = g[13 * LXYZ + idx];
        g_temp[23 * LXYZ + idx] = g[21 * LXYZ + idx];
        g_temp[26 * LXYZ + idx] = g[20 * LXYZ + idx];
    }
    if (ix == LX-1) {  // Right wall (adiabatic)
        g_temp[3  * LXYZ + idx] = g[     LXYZ + idx];
        g_temp[9  * LXYZ + idx] = g[7  * LXYZ + idx];
        g_temp[8  * LXYZ + idx] = g[10 * LXYZ + idx];
        g_temp[17 * LXYZ + idx] = g[11 * LXYZ + idx];
        g_temp[25 * LXYZ + idx] = g[19 * LXYZ + idx];
        g_temp[24 * LXYZ + idx] = g[22 * LXYZ + idx];
        g_temp[13 * LXYZ + idx] = g[15 * LXYZ + idx];
        g_temp[21 * LXYZ + idx] = g[23 * LXYZ + idx];
        g_temp[20 * LXYZ + idx] = g[26 * LXYZ + idx];
    }
    if (iy == 0) {  // Bottom wall (hot plate)
        double density = rho[idx];
        g_temp[2  * LXYZ + idx] = -g[4  * LXYZ + idx] + 2.0 * d_tp[2]  * d_tHot * (1.0 + density);
        g_temp[7  * LXYZ + idx] = -g[9  * LXYZ + idx] + 2.0 * d_tp[7]  * d_tHot * (1.0 + density);
        g_temp[8  * LXYZ + idx] = -g[10 * LXYZ + idx] + 2.0 * d_tp[8]  * d_tHot * (1.0 + density);
        g_temp[12 * LXYZ + idx] = -g[18 * LXYZ + idx] + 2.0 * d_tp[12] * d_tHot * (1.0 + density);
        g_temp[19 * LXYZ + idx] = -g[25 * LXYZ + idx] + 2.0 * d_tp[19] * d_tHot * (1.0 + density);
        g_temp[20 * LXYZ + idx] = -g[26 * LXYZ + idx] + 2.0 * d_tp[20] * d_tHot * (1.0 + density);
        g_temp[16 * LXYZ + idx] = -g[14 * LXYZ + idx] + 2.0 * d_tp[16] * d_tHot * (1.0 + density);
        g_temp[23 * LXYZ + idx] = -g[21 * LXYZ + idx] + 2.0 * d_tp[23] * d_tHot * (1.0 + density);
        g_temp[24 * LXYZ + idx] = -g[22 * LXYZ + idx] + 2.0 * d_tp[24] * d_tHot * (1.0 + density);
    }
    if (iy == LY-1) {  // Top wall (cold plate)
        double density = rho[idx];
        g_temp[4  * LXYZ + idx] = -g[2  * LXYZ + idx] + 2.0 * d_tp[4]  * d_tCold * (1.0 + density);
        g_temp[9  * LXYZ + idx] = -g[7  * LXYZ + idx] + 2.0 * d_tp[9]  * d_tCold * (1.0 + density);
        g_temp[10 * LXYZ + idx] = -g[8  * LXYZ + idx] + 2.0 * d_tp[10] * d_tCold * (1.0 + density);
        g_temp[18 * LXYZ + idx] = -g[12 * LXYZ + idx] + 2.0 * d_tp[18] * d_tCold * (1.0 + density);
        g_temp[25 * LXYZ + idx] = -g[19 * LXYZ + idx] + 2.0 * d_tp[25] * d_tCold * (1.0 + density);
        g_temp[26 * LXYZ + idx] = -g[20 * LXYZ + idx] + 2.0 * d_tp[26] * d_tCold * (1.0 + density);
        g_temp[14 * LXYZ + idx] = -g[16 * LXYZ + idx] + 2.0 * d_tp[14] * d_tCold * (1.0 + density);
        g_temp[21 * LXYZ + idx] = -g[23 * LXYZ + idx] + 2.0 * d_tp[21] * d_tCold * (1.0 + density);
        g_temp[22 * LXYZ + idx] = -g[24 * LXYZ + idx] + 2.0 * d_tp[22] * d_tCold * (1.0 + density);
    }
    if (iz == 0) {  // Front wall (adiabatic)
        g_temp[5  * LXYZ + idx] = g[6  * LXYZ + idx];
        g_temp[11 * LXYZ + idx] = g[17 * LXYZ + idx];
        g_temp[12 * LXYZ + idx] = g[18 * LXYZ + idx];
        g_temp[13 * LXYZ + idx] = g[15 * LXYZ + idx];
        g_temp[14 * LXYZ + idx] = g[16 * LXYZ + idx];
        g_temp[19 * LXYZ + idx] = g[25 * LXYZ + idx];
        g_temp[20 * LXYZ + idx] = g[26 * LXYZ + idx];
        g_temp[21 * LXYZ + idx] = g[23 * LXYZ + idx];
        g_temp[22 * LXYZ + idx] = g[24 * LXYZ + idx];
    }
    if (iz == LZ-1) {  // Back wall (adiabatic)
        g_temp[6  * LXYZ + idx] = g[5  * LXYZ + idx];
        g_temp[17 * LXYZ + idx] = g[11 * LXYZ + idx];
        g_temp[18 * LXYZ + idx] = g[12 * LXYZ + idx];
        g_temp[15 * LXYZ + idx] = g[13 * LXYZ + idx];
        g_temp[16 * LXYZ + idx] = g[14 * LXYZ + idx];
        g_temp[25 * LXYZ + idx] = g[19 * LXYZ + idx];
        g_temp[26 * LXYZ + idx] = g[20 * LXYZ + idx];
        g_temp[23 * LXYZ + idx] = g[21 * LXYZ + idx];
        g_temp[24 * LXYZ + idx] = g[22 * LXYZ + idx];
    }
}