#ifndef PARAMETERS_H
#define PARAMETERS_H

// Continue computation
#define CONTINUE_STEP 0

// Grid dimensions (全局网格大小)
#define LX 400
#define LY 400
#define LZ 400
#define LXYZ (LX * LY * LZ)
#define NPOP 27

// Multi-GPU configuration
#define MAX_GPUS 8              // 最大支持GPU数量
#define HALO_WIDTH 1            // Halo区域宽度(层数)

// Time steps
#define NEND 1000
#define NDIAG 10000
#define NFLOWOUT 10000
#define NNUOUT 10000

// Physical parameters
#define RAYLEIGH 1.0e7
#define PRANDTL 0.71
#define T_HOT 1.0
#define T_COLD 0.0
#define GRAVITY 1.0
#define BETA (0.1 / (LY * 5.0))

// Block dimensions for CUDA kernels
#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 4

// 计算宏 - 多GPU环境下的局部网格计算
#define GET_LOCAL_SIZE(ly_local_with_halo) (LX * (ly_local_with_halo) * LZ)
#define GET_BUFFER_SIZE() (LX * LZ * NPOP)
#define GET_LY_LOCAL(num_gpus) (LY / (num_gpus))
#define GET_LY_LOCAL_WITH_HALO(num_gpus) (GET_LY_LOCAL(num_gpus) + 2 * HALO_WIDTH)

#endif 