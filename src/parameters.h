#ifndef PARAMETERS_H
#define PARAMETERS_H

// Continue computation
#define CONTINUE_STEP 0

// Grid dimensions
#define LX 400
#define LY 400
#define LZ 400
#define LXYZ (LX * LY * LZ)
#define NPOP 27

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

#endif 