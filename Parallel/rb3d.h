#ifndef RB3D_H
#define RB3D_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>

#include "parameters.h"

// Device constants declaration
extern __constant__ int d_cix[NPOP];
extern __constant__ int d_ciy[NPOP];
extern __constant__ int d_ciz[NPOP];
extern __constant__ double d_tp[NPOP];
extern __constant__ double d_tau;
extern __constant__ double d_tauc;
extern __constant__ double d_grav0;
extern __constant__ double d_beta;
extern __constant__ double d_tHot;
extern __constant__ double d_tCold;
extern __constant__ double d_t0;
extern __constant__ double d_diff;
extern __constant__ double d_visc;

// Global variables declaration
extern double rayl, prand;
extern double visc, tau, u0, diff, tauc, tauci, grav0, beta;
extern double tHot, tCold, t0;
extern int continue_step;

// Device pointers declaration
extern double *d_f, *d_g, *d_rho, *d_ux, *d_uy, *d_uz, *d_phi;
extern double *d_force_realx, *d_force_realy, *d_force_realz;
extern double *d_f_temp, *d_g_temp;

// Host pointers declaration
extern double *h_rho, *h_ux, *h_uy, *h_uz, *h_phi;
extern double *h_f, *h_g;

// Function declarations
void initialize();
void initialize_parallel(int num_gpus);
void initialize_constants();
void initialize_arrays_parallel(int num_gpus);
void initialize_from_file_parallel(int num_gpus);

__global__ void init_arrays(double *f, double *g, double *rho, double *ux, double *uy, double *uz, 
                           double *phi, double *force_realx, double *force_realy, double *force_realz);

__global__ void collision_BGK(double *f, double *force_realx, double *force_realy, double *force_realz, 
                             double *rho, double *ux, double *uy, double *uz);
__global__ void collision_BGK_scalar(double *g, double *force_realx, double *force_realy, double *force_realz, 
                                    double *rho, double *ux, double *uy, double *uz, double *phi);

__global__ void streaming(double *f, double *f_temp);
__global__ void streaming_scalar(double *g, double *g_temp, double *rho);

__global__ void macrovar(double *f, double *g, double *force_realx, double *force_realy, double *force_realz, 
                        double *rho, double *ux, double *uy, double *uz, double *phi);
__global__ void forcing(double *force_realx, double *force_realy, double *force_realz, double *phi);

void diag_flow(int istep);
void output_flow(int istep);
void output_nu(int istep);
void output_profile(int istep);
void output_fg(int istep);

// 多GPU输出函数声明
void diag_flow_parallel(int istep, int num_gpus);
void output_flow_parallel(int istep, int num_gpus);
void output_nu_parallel(int istep, int num_gpus);
void output_profile_parallel(int istep, int num_gpus);
void output_fg_parallel(int istep, int num_gpus);

#endif 