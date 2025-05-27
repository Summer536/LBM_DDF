#include "rb3d.h"

int main() {
    struct timeval start_time, end_time;
    double cpu_time_used = 0.0, gpu_time_used = 0.0;
    cudaEvent_t gpu_start, gpu_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_stop));
    gettimeofday(&start_time, NULL);

    initialize();

    CHECK_CUDA_ERROR(cudaMemcpy(h_rho, d_rho, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ux,  d_ux,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uy,  d_uy,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uz,  d_uz,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_phi, d_phi, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));

    dim3 grid((LX + BLOCK_X - 1) / BLOCK_X, (LY + BLOCK_Y - 1) / BLOCK_Y, (LZ + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    #pragma unroll
    for (int istep = continue_step + 1; istep <= NEND; istep++) {
        // GPU timing start
        CHECK_CUDA_ERROR(cudaEventRecord(gpu_start));

        forcing<<<grid, block>>>(d_force_realx, d_force_realy, d_force_realz, d_phi);

        collision_BGK<<<grid, block>>>(d_f, d_force_realx, d_force_realy, d_force_realz, 
                                     d_rho, d_ux, d_uy, d_uz);
        // collision_BGK_scalar<<<grid, block, 2 * NPOP * sizeof(double)>>>(d_g, d_force_realx, d_force_realy, d_force_realz, 
        //                                                         d_rho, d_ux, d_uy, d_uz, d_phi, d_Tx, d_Ty, d_Tz);
        collision_BGK_scalar<<<grid, block>>>(d_g, d_force_realx, d_force_realy, d_force_realz, 
                                      d_rho, d_ux, d_uy, d_uz, d_phi);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        streaming<<<grid, block>>>(d_f, d_f_temp);
        CHECK_CUDA_ERROR(cudaMemcpy(d_f, d_f_temp, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToDevice));
        streaming_scalar<<<grid, block>>>(d_g, d_g_temp, d_rho);
        CHECK_CUDA_ERROR(cudaMemcpy(d_g, d_g_temp, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        macrovar<<<grid, block>>>(d_f, d_g, d_force_realx, d_force_realy, d_force_realz, 
                                 d_rho, d_ux, d_uy, d_uz, d_phi);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // GPU timing end
        CHECK_CUDA_ERROR(cudaEventRecord(gpu_stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(gpu_stop));

        float gpu_milliseconds = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop));
        gpu_time_used += (double)gpu_milliseconds / 1000.0;

        // CPU timing start
        clock_t cpu_start = clock();

        char dirname[256];
        snprintf(dirname, sizeof(dirname), "RB_Ra%.2e_Pr%.2f", rayl, prand);

        if (istep % NDIAG == 0) {
            diag_flow(istep);
        }
        if (istep % NFLOWOUT == 0 || istep == NEND) {
            output_flow(istep);
        }
        if (istep % NNUOUT == 0) {
            output_nu(istep);
            output_profile(istep);
        }
        if (istep == NEND) {
            output_fg(istep);
        }

        // CPU timing end
        cpu_time_used += (double)(clock() - cpu_start) / CLOCKS_PER_SEC;
    }

    // calculate total time
    gettimeofday(&end_time, NULL);
    double total_time = (end_time.tv_sec - start_time.tv_sec) + 
                       (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    printf("Simulation completed.\n");
    printf("Rayleigh number: %.2e\n", rayl);
    printf("Prandtl number: %.2f\n", prand);
    printf("Total iterations: %d\n", NEND);
    printf("CPU computation time: %.2f seconds\n", cpu_time_used);
    printf("GPU computation time: %.2f seconds\n", gpu_time_used);
    printf("Total computation time: %.2f seconds\n", total_time);

    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_stop));

    CHECK_CUDA_ERROR(cudaFree(d_f));
    CHECK_CUDA_ERROR(cudaFree(d_g));
    CHECK_CUDA_ERROR(cudaFree(d_rho));
    CHECK_CUDA_ERROR(cudaFree(d_ux));
    CHECK_CUDA_ERROR(cudaFree(d_uy));
    CHECK_CUDA_ERROR(cudaFree(d_uz));
    CHECK_CUDA_ERROR(cudaFree(d_phi));
    CHECK_CUDA_ERROR(cudaFree(d_force_realx));
    CHECK_CUDA_ERROR(cudaFree(d_force_realy));
    CHECK_CUDA_ERROR(cudaFree(d_force_realz));
    CHECK_CUDA_ERROR(cudaFree(d_f_temp));
    CHECK_CUDA_ERROR(cudaFree(d_g_temp));

    free(h_f);
    free(h_g);
    free(h_rho);
    free(h_ux);
    free(h_uy);
    free(h_uz);
    free(h_phi);

    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}