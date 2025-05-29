#include "multi_gpu.h"
#include "rb3d.h"

int main() {
    struct timeval start_time, end_time;
    double cpu_time_used = 0.0, gpu_time_used = 0.0;
    
    // GPU事件计时器（每个GPU一个）
    cudaEvent_t **gpu_start = NULL, **gpu_stop = NULL;
    
    gettimeofday(&start_time, NULL);

    // 确定GPU数量（这里可以从命令行参数或环境变量获取）
    int num_gpus = 1; // 默认单GPU，实际使用时可以修改
    char *gpu_env = getenv("NUM_GPUS");
    if (gpu_env != NULL) {
        num_gpus = atoi(gpu_env);
        if (num_gpus <= 0 || num_gpus > 8) {
            fprintf(stderr, "Invalid number of GPUs: %d. Using 1 GPU.\n", num_gpus);
            num_gpus = 1;
        }
    }
    
    printf("Starting simulation with %d GPU(s)\n", num_gpus);

    // 多GPU初始化
    initialize_parallel(num_gpus);

    // 为每个GPU创建计时事件
    gpu_start = (cudaEvent_t**)malloc(num_gpus * sizeof(cudaEvent_t*));
    gpu_stop = (cudaEvent_t**)malloc(num_gpus * sizeof(cudaEvent_t*));
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        gpu_start[gpu_id] = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
        gpu_stop[gpu_id] = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
        CHECK_CUDA_ERROR(cudaEventCreate(gpu_start[gpu_id]));
        CHECK_CUDA_ERROR(cudaEventCreate(gpu_stop[gpu_id]));
    }

    printf("Multi-GPU simulation starting...\n");
    printf("Grid: %d x %d x %d, GPUs: %d\n", LX, LY, LZ, num_gpus);
    printf("Local Y per GPU: %d (+%d halo)\n", GET_LY_LOCAL(num_gpus), 2 * HALO_WIDTH);

    #pragma unroll
    for (int istep = continue_step + 1; istep <= NEND; istep++) {
        for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
            CHECK_CUDA_ERROR(cudaEventRecord(*gpu_start[gpu_id]));
        }

        // 1. 强制项和宏观变量计算
        macrovar_step_parallel(num_gpus);
        
        // 2. 碰撞步
        collision_step_parallel(num_gpus);
        
        // 3. 流步（包含halo交换）
        streaming_step_parallel(num_gpus);
        
        for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        // GPU计时结束
        for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
            CHECK_CUDA_ERROR(cudaEventRecord(*gpu_stop[gpu_id]));
            CHECK_CUDA_ERROR(cudaEventSynchronize(*gpu_stop[gpu_id]));
            
            float gpu_milliseconds = 0.0f;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_milliseconds, *gpu_start[gpu_id], *gpu_stop[gpu_id]));
            gpu_time_used += (double)gpu_milliseconds / 1000.0 / num_gpus; // 平均GPU时间
        }

        // CPU计时开始
        clock_t cpu_start = clock();

        if (istep % NDIAG == 0) {
            diag_flow_parallel(istep, num_gpus);
        }
        if (istep % NFLOWOUT == 0 || istep == NEND) {
            output_flow_parallel(istep, num_gpus);
        }
        if (istep % NNUOUT == 0) {
            output_nu_parallel(istep, num_gpus);
            output_profile_parallel(istep, num_gpus);
        }
        if (istep == NEND) {
            output_fg_parallel(istep, num_gpus);
        }

        cpu_time_used += (double)(clock() - cpu_start) / CLOCKS_PER_SEC;
        
        if (istep % 100 == 0) {
            printf("Step %d/%d completed\n", istep, NEND);
        }
    }

    gettimeofday(&end_time, NULL);
    double total_time = (end_time.tv_sec - start_time.tv_sec) + 
                       (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    printf("Simulation completed.\n");
    printf("Rayleigh number: %.2e\n", rayl);
    printf("Prandtl number: %.2f\n", prand);
    printf("Total iterations: %d\n", NEND);
    printf("Number of GPUs: %d\n", num_gpus);
    printf("CPU computation time: %.2f seconds\n", cpu_time_used);
    printf("GPU computation time: %.2f seconds (per GPU average)\n", gpu_time_used);
    printf("Total computation time: %.2f seconds\n", total_time);
    printf("Performance: %.2f MLUPS\n", (double)NEND * LXYZ / (total_time * 1e6));

    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        CHECK_CUDA_ERROR(cudaEventDestroy(*gpu_start[gpu_id]));
        CHECK_CUDA_ERROR(cudaEventDestroy(*gpu_stop[gpu_id]));
        free(gpu_start[gpu_id]);
        free(gpu_stop[gpu_id]);
    }
    free(gpu_start);
    free(gpu_stop);

    multi_gpu_finalize();

    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        CHECK_CUDA_ERROR(cudaDeviceReset());
    }

    return 0;
} 