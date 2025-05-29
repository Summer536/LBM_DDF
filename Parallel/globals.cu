#include "rb3d.h"
#include "parameters.h"

__constant__ int d_cix[NPOP];
__constant__ int d_ciy[NPOP];
__constant__ int d_ciz[NPOP];
__constant__ double d_tp[NPOP];
__constant__ double d_tau, d_tauc, d_grav0, d_beta, d_tHot, d_tCold, d_t0;
__constant__ double d_diff, d_visc;

double rayl = RAYLEIGH;
double prand = PRANDTL;
double visc = (double)LY * sqrt(grav0 * beta * (tHot - tCold) * (double)LY * prand / rayl); 
double tau = 3.0 * visc + 0.5;
double u0 = sqrt(grav0 * beta * (tHot - tCold) * (double)LY);
double diff = visc / prand;
double tauc = 3.0 * diff + 0.5;
double tauci = 1.0 / tauc;
double grav0 = GRAVITY;
double beta = BETA;
double tHot = T_HOT;
double tCold = T_COLD;
double t0 = 0.5 * (tCold + tHot);

int continue_step = CONTINUE_STEP;

// Device pointers definition
double *d_f = nullptr, *d_g = nullptr, *d_rho = nullptr, *d_ux = nullptr, *d_uy = nullptr, *d_uz = nullptr, *d_phi = nullptr;
double *d_force_realx = nullptr, *d_force_realy = nullptr, *d_force_realz = nullptr;
double *d_f_temp = nullptr, *d_g_temp = nullptr;

// Host pointers definition
double *h_rho = nullptr, *h_ux = nullptr, *h_uy = nullptr, *h_uz = nullptr, *h_phi = nullptr;
double *h_f = nullptr, *h_g = nullptr;