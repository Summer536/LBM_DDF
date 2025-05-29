#include "rb3d.h"
#include "multi_gpu.h"

// ===================== 多GPU并行输出模块 =====================

// 多GPU统计输出函数
void diag_flow_parallel(int istep, int num_gpus) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        #ifdef _WIN32
            _mkdir(dirname);
        #else
            mkdir(dirname, 0700);
        #endif
        printf("Created directory: %s\n", dirname);
    }

    // 计算全局统计量（需要跨GPU归约）
    double global_umean = 0.0, global_vmean = 0.0, global_wmean = 0.0, global_tmean = 0.0;
    long long global_count = 0;

    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        GPU_Domain *domain = gpu_manager->domains[gpu_id];
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        
        // 将GPU数据复制到主机
        size_t local_field_size = domain->local_size * sizeof(double);
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_ux_local, domain->d_ux_local, local_field_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_uy_local, domain->d_uy_local, local_field_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_uz_local, domain->d_uz_local, local_field_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_phi_local, domain->d_phi_local, local_field_size, cudaMemcpyDeviceToHost));
        
        // 计算局部统计量（只统计非halo区域）
        for (int iz = 0; iz < LZ; iz++) {
            for (int iy_local = HALO_WIDTH; iy_local < domain->LY_local_with_halo - HALO_WIDTH; iy_local++) {
                for (int ix = 0; ix < LX; ix++) {
                    int idx_local = get_local_index(ix, iy_local, iz, domain->LY_local_with_halo);
                    
                    global_umean += domain->h_ux_local[idx_local];
                    global_vmean += domain->h_uy_local[idx_local];
                    global_wmean += domain->h_uz_local[idx_local];
                    global_tmean += domain->h_phi_local[idx_local];
                    global_count++;
                }
            }
        }
    }

    // 计算全局平均值
    global_umean /= global_count;
    global_vmean /= global_count;
    global_wmean /= global_count;
    global_tmean /= global_count;

    // 输出到控制台（只在主GPU打印）
    printf("Step %-7d umean = %14.6e, vmean = %14.6e, wmean = %14.6e, tmean = %14.6e\n", 
            istep, global_umean, global_vmean, global_wmean, global_tmean);
    fflush(stdout);

    // 输出到统计文件
    char stat_filename[512];
    snprintf(stat_filename, sizeof(stat_filename), "%s/statistics.dat", dirname);
    FILE *f_stat = fopen(stat_filename, "a");
    if (f_stat == NULL) {
        fprintf(stderr, "Error opening file: %s\n", stat_filename);
        exit(EXIT_FAILURE);
    }

    int write_status = fprintf(f_stat, "%-7d %14.6e %14.6e %14.6e %14.6e\n", 
                             istep, global_umean, global_vmean, global_wmean, global_tmean);
    if (write_status < 0) {
        fprintf(stderr, "Error writing to statistics file\n");
        fclose(f_stat);
        exit(EXIT_FAILURE);
    }

    if (fflush(f_stat) != 0) {
        fprintf(stderr, "Error flushing statistics file\n");
        fclose(f_stat);
        exit(EXIT_FAILURE);
    }

    fclose(f_stat);
}

// 多GPU流场数据输出函数
void output_flow_parallel(int istep, int num_gpus) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        #ifdef _WIN32
            _mkdir(dirname);
        #else
            mkdir(dirname, 0700);
        #endif
        printf("Created directory: %s\n", dirname);
    }

    char data_filename[512];
    snprintf(data_filename, sizeof(data_filename), "%s/%09d.dat", dirname, istep);

    FILE *f_data = fopen(data_filename, "wb");
    if (f_data == NULL) {
        fprintf(stderr, "Cannot open file: %s\n", data_filename);
        exit(EXIT_FAILURE);
    }

    // 按照全局Y坐标顺序，逐GPU输出数据
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        GPU_Domain *domain = gpu_manager->domains[gpu_id];
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        
        // 将GPU数据复制到主机
        size_t local_field_size = domain->local_size * sizeof(double);
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_rho_local, domain->d_rho_local, local_field_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_ux_local, domain->d_ux_local, local_field_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_uy_local, domain->d_uy_local, local_field_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_uz_local, domain->d_uz_local, local_field_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_phi_local, domain->d_phi_local, local_field_size, cudaMemcpyDeviceToHost));

        // 重构数据并按全局顺序写入（只写非halo区域）
        // 先分配临时全局数组用于重构
        if (gpu_id == 0) {
            // 分配全局数组并写入文件头（第一个GPU时）
            // 这里我们采用逐GPU写入的策略，避免大量内存消耗
        }

        // 写入当前GPU的数据（按全局坐标顺序）
        write_gpu_data_to_file(f_data, domain, "rho");
    }

    // 重复上述过程为其他变量（ux, uy, uz, phi）
    for (int var_idx = 1; var_idx < 5; var_idx++) {
        const char* var_names[] = {"rho", "ux", "uy", "uz", "phi"};
        for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
            GPU_Domain *domain = gpu_manager->domains[gpu_id];
            write_gpu_data_to_file(f_data, domain, var_names[var_idx]);
        }
    }

    if (fflush(f_data) != 0) {
        fprintf(stderr, "Error flushing output file\n");
        fclose(f_data);
        exit(EXIT_FAILURE);
    }
    
    fclose(f_data);
}

// 辅助函数：将单个GPU的数据写入文件
void write_gpu_data_to_file(FILE *f_data, GPU_Domain *domain, const char *var_name) {
    double *data_ptr = NULL;
    
    // 选择对应的数据指针
    if (strcmp(var_name, "rho") == 0) {
        data_ptr = domain->h_rho_local;
    } else if (strcmp(var_name, "ux") == 0) {
        data_ptr = domain->h_ux_local;
    } else if (strcmp(var_name, "uy") == 0) {
        data_ptr = domain->h_uy_local;
    } else if (strcmp(var_name, "uz") == 0) {
        data_ptr = domain->h_uz_local;
    } else if (strcmp(var_name, "phi") == 0) {
        data_ptr = domain->h_phi_local;
    } else {
        fprintf(stderr, "Unknown variable name: %s\n", var_name);
        exit(EXIT_FAILURE);
    }

    // 按全局坐标顺序写入（只写非halo区域）
    for (int iz = 0; iz < LZ; iz++) {
        for (int iy_local = HALO_WIDTH; iy_local < domain->LY_local_with_halo - HALO_WIDTH; iy_local++) {
            for (int ix = 0; ix < LX; ix++) {
                int idx_local = get_local_index(ix, iy_local, iz, domain->LY_local_with_halo);
                
                size_t written = fwrite(&data_ptr[idx_local], sizeof(double), 1, f_data);
                if (written != 1) {
                    fprintf(stderr, "Error writing %s data at GPU %d, local (%d,%d,%d)\n", 
                            var_name, domain->gpu_id, ix, iy_local, iz);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
}

void diag_flow(int istep) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        #ifdef _WIN32
            _mkdir(dirname);
        #else
            mkdir(dirname, 0700);
        #endif
        printf("Created directory: %s\n", dirname);
    }

    // Copy data from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_ux,  d_ux,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uy,  d_uy,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uz,  d_uz,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_phi, d_phi, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));

    // Calculate statistics
    double umean = 0.0, vmean = 0.0, wmean = 0.0, tmean = 0.0;
    for (int i = 0; i < LXYZ; i++) {
        umean += h_ux[i];
        vmean += h_uy[i];
        wmean += h_uz[i];
        tmean += h_phi[i];
    }
    umean /= LXYZ;
    vmean /= LXYZ;
    wmean /= LXYZ;
    tmean /= LXYZ;

    // Output to console
    printf("Step %-7d umean = %14.6e, vmean = %14.6e, wmean = %14.6e, tmean = %14.6e\n", 
            istep, umean, vmean, wmean, tmean);
    fflush(stdout);

    // Output to statistics file
    char stat_filename[512];
    snprintf(stat_filename, sizeof(stat_filename), "%s/statistics.dat", dirname);
    FILE *f_stat = fopen(stat_filename, "a");
    if (f_stat == NULL) {
        fprintf(stderr, "Error opening file: %s\n", stat_filename);
        exit(EXIT_FAILURE);
    }

    int write_status = fprintf(f_stat, "%-7d %14.6e %14.6e %14.6e %14.6e\n", 
                             istep, umean, vmean, wmean, tmean);
    if (write_status < 0) {
        fprintf(stderr, "Error writing to statistics file\n");
        fclose(f_stat);
        exit(EXIT_FAILURE);
    }

    if (fflush(f_stat) != 0) {
        fprintf(stderr, "Error flushing statistics file\n");
        fclose(f_stat);
        exit(EXIT_FAILURE);
    }

    fclose(f_stat);
}

void output_flow(int istep) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        #ifdef _WIN32
            _mkdir(dirname);
        #else
            mkdir(dirname, 0700);
        #endif
        printf("Created directory: %s\n", dirname);
    }

    // Copy data from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_rho, d_rho, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ux,  d_ux,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uy,  d_uy,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uz,  d_uz,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_phi, d_phi, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));

    // Output flow field data to binary file
    char data_filename[512];
    snprintf(data_filename, sizeof(data_filename), "%s/%09d.dat", dirname, istep);

    FILE *f_data = fopen(data_filename, "wb");
    if (f_data == NULL) {
        fprintf(stderr, "Cannot open file: %s\n", data_filename);
        exit(EXIT_FAILURE);
    }

    size_t written_elements;
    written_elements = fwrite(h_rho, sizeof(double), LXYZ, f_data);
    if (written_elements != LXYZ) {
        fprintf(stderr, "Error writing h_rho data: wrote %zu of %d elements\n", 
                written_elements, LXYZ);
        fclose(f_data);
        exit(EXIT_FAILURE);
    }
    
    written_elements = fwrite(h_ux, sizeof(double), LXYZ, f_data);
    if (written_elements != LXYZ) {
        fprintf(stderr, "Error writing h_ux data: wrote %zu of %d elements\n", 
                written_elements, LXYZ);
        fclose(f_data);
        exit(EXIT_FAILURE);
    }
    
    written_elements = fwrite(h_uy, sizeof(double), LXYZ, f_data);
    if (written_elements != LXYZ) {
        fprintf(stderr, "Error writing h_uy data: wrote %zu of %d elements\n", 
                written_elements, LXYZ);
        fclose(f_data);
        exit(EXIT_FAILURE);
    }
    
    written_elements = fwrite(h_uz, sizeof(double), LXYZ, f_data);
    if (written_elements != LXYZ) {
        fprintf(stderr, "Error writing h_uz data: wrote %zu of %d elements\n", 
                written_elements, LXYZ);
        fclose(f_data);
        exit(EXIT_FAILURE);
    }
    
    written_elements = fwrite(h_phi, sizeof(double), LXYZ, f_data);
    if (written_elements != LXYZ) {
        fprintf(stderr, "Error writing h_phi data: wrote %zu of %d elements\n", 
                written_elements, LXYZ);
        fclose(f_data);
        exit(EXIT_FAILURE);
    }
    
    if (fflush(f_data) != 0) {
        fprintf(stderr, "Error flushing output file\n");
        fclose(f_data);
        exit(EXIT_FAILURE);
    }
    
    fclose(f_data);
}

void output_nu(int istep) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        #ifdef _WIN32
            _mkdir(dirname);
        #else
            mkdir(dirname, 0700);
        #endif
        printf("Created directory: %s\n", dirname);
    }

    char filename[512]; 

    if (rayl == 1e5) {
        snprintf(filename, sizeof(filename), "%s/Nu_1e50.txt", dirname);
    } else if (rayl == 5e5) {
        snprintf(filename, sizeof(filename), "%s/Nu_5e50.txt", dirname);
    } else if (rayl == 1e6) {
        snprintf(filename, sizeof(filename), "%s/Nu_1e60.txt", dirname);
    } else if (rayl == 5e6) {
        snprintf(filename, sizeof(filename), "%s/Nu_5e60.txt", dirname);
    } else if (rayl == 1e7) {
        snprintf(filename, sizeof(filename), "%s/Nu_1e70.txt", dirname);
    } else if (rayl == 1e8) {
        snprintf(filename, sizeof(filename), "%s/Nu_1e80.txt", dirname);
    } else {
        snprintf(filename, sizeof(filename), "%s/Nu_%.2e.txt", dirname, rayl);
    }

    FILE *f_nu = fopen(filename, "a");
    if (f_nu == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    CHECK_CUDA_ERROR(cudaMemcpy(h_uy,  d_uy,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_phi, d_phi, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));

    double *h_uyT = (double*)malloc(LXYZ * sizeof(double));
    double *h_Ty  = (double*)malloc(LXYZ * sizeof(double));
    if (h_uyT == NULL || h_Ty == NULL) {
        fprintf(stderr, "Failed to allocate memory for Nusselt number calculation\n");
        free(h_uyT);
        free(h_Ty);
        fclose(f_nu);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < LXYZ; i++) {
        h_uyT[i] = h_uy[i] * h_phi[i];
    }

    for (int iz = 0; iz < LZ; iz++) {
        for (int ix = 0; ix < LX; ix++) {
            for (int iy = 0; iy < LY; iy++) {
                int idx = (iz * LY + iy) * LX + ix;
                
                if (iy == 0) {  
                    int idx_p1 = (iz * LY + (iy+1)) * LX + ix;
                    int idx_p2 = (iz * LY + (iy+2)) * LX + ix;
                    h_Ty[idx] = (-3.0*h_phi[idx] + 4.0*h_phi[idx_p1] - h_phi[idx_p2]) / 2.0;
                }
                else if (iy == LY-1) {  
                    int idx_m1 = (iz * LY + (iy-1)) * LX + ix;
                    int idx_m2 = (iz * LY + (iy-2)) * LX + ix;
                    h_Ty[idx] = (3.0*h_phi[idx] - 4.0*h_phi[idx_m1] + h_phi[idx_m2]) / 2.0;
                }
                else {  
                    int idx_p1 = (iz * LY + (iy+1)) * LX + ix;
                    int idx_m1 = (iz * LY + (iy-1)) * LX + ix;
                    h_Ty[idx] = (h_phi[idx_p1] - h_phi[idx_m1]) / 2.0;
                }
            }
        }
    }

    // Compute Nu_bulk
    double uyT_avg = 0.0;
    for (int i = 0; i < LXYZ; i++) {
        uyT_avg += h_uyT[i];
    }
    uyT_avg /= LXYZ;
    double Nu_bulk = 1.0 + uyT_avg * LY / (diff * (tHot - tCold));

    // Compute Nu_top and Nu_bottom using local temperature gradients
    double dTdy_top = 0.0, dTdy_bottom = 0.0;
    int count_top = 0, count_bottom = 0;
    
    for (int ix = 0; ix < LX; ix++) {
        for (int iz = 0; iz < LZ; iz++) {
            int idx_top = (iz * LY + (LY-1)) * LX + ix;
            int idx_bottom = iz * LY * LX + ix;
            
            dTdy_top += h_Ty[idx_top];
            dTdy_bottom += h_Ty[idx_bottom];
            
            count_top++;
            count_bottom++;
        }
    }
    
    dTdy_top /= count_top;
    dTdy_bottom /= count_bottom;
    
    double Nu_top = -dTdy_top * LY / (tHot - tCold);
    double Nu_bottom = -dTdy_bottom * LY / (tHot - tCold);

    // Output to file
    int write_status = fprintf(f_nu, "%15.6E%15.6E%15.6E%15.6E%15.6E\n", 
                             (double)istep, 0.0, Nu_bulk, Nu_top, Nu_bottom);
    if (write_status < 0) {
        fprintf(stderr, "Error writing to Nu file\n");
        free(h_uyT);
        free(h_Ty);
        fclose(f_nu);
        exit(EXIT_FAILURE);
    }
    
    if (fflush(f_nu) != 0) {
        fprintf(stderr, "Error flushing Nu file\n");
        free(h_uyT);
        free(h_Ty);
        fclose(f_nu);
        exit(EXIT_FAILURE);
    }
    
    fclose(f_nu);
    free(h_uyT);
    free(h_Ty);
}

void output_profile(int istep) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        #ifdef _WIN32
            _mkdir(dirname);
        #else
            mkdir(dirname, 0700);
        #endif
        printf("Created directory: %s\n", dirname);
    }

    char filename[512]; 

    if (rayl == 1e5) {
        snprintf(filename, sizeof(filename), "%s/Profile_1e50.txt", dirname);
    } else if (rayl == 5e5) {
        snprintf(filename, sizeof(filename), "%s/Profile_5e50.txt", dirname);
    } else if (rayl == 1e6) {
        snprintf(filename, sizeof(filename), "%s/Profile_1e60.txt", dirname);
    } else if (rayl == 5e6) {
        snprintf(filename, sizeof(filename), "%s/Profile_5e60.txt", dirname);
    } else if (rayl == 1e7) {
        snprintf(filename, sizeof(filename), "%s/Profile_1e70.txt", dirname);
    } else if (rayl == 1e8) {
        snprintf(filename, sizeof(filename), "%s/Profile_1e80.txt", dirname);
    } else {
        snprintf(filename, sizeof(filename), "%s/Profile_%.2e.txt", dirname, rayl);
    }

    FILE *f_profile = fopen(filename, "a");
    if (f_profile == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Copy data from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_rho, d_rho, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_ux,  d_ux,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uy,  d_uy,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_uz,  d_uz,  LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_phi, d_phi, LXYZ * sizeof(double), cudaMemcpyDeviceToHost));

    // Write the current step to the file
    int write_status = fprintf(f_profile, "%d\n", istep);
    if (write_status < 0) {
        fprintf(stderr, "Error writing step number to profile file\n");
        fclose(f_profile);
        exit(EXIT_FAILURE);
    }

    // Loop over y-planes
    for (int iy = 0; iy < LY; iy++) {
        double sum_ux = 0.0, sum_uy = 0.0, sum_uz = 0.0, sum_phi = 0.0;
        double sum_phi_sq = 0.0;
        int count = 0;

        for (int ix = 0; ix < LX; ix++) {
            for (int iz = 0; iz < LZ; iz++) {
                int idx = (iz * LY + iy) * LX + ix;

                sum_ux    += h_ux[idx];
                sum_uy    += h_uy[idx];
                sum_uz    += h_uz[idx];
                sum_phi   += h_phi[idx];
                sum_phi_sq += h_phi[idx] * h_phi[idx];
                count++;
            }
        }

        double avg_ux  = sum_ux  / count;
        double avg_uy  = sum_uy  / count;
        double avg_uz  = sum_uz  / count;
        double avg_phi = sum_phi / count;
        double phi_rms = sqrt(sum_phi_sq / count - avg_phi * avg_phi);

        // Write the averages and phi_rms to the file
        write_status = fprintf(f_profile, "%14.6e %14.6e %14.6e %14.6e %14.6e\n",
                             avg_ux, avg_uy, avg_uz, avg_phi, phi_rms);
        if (write_status < 0) {
            fprintf(stderr, "Error writing profile data\n");
            fclose(f_profile);
            exit(EXIT_FAILURE);
        }
    }

    if (fflush(f_profile) != 0) {
        fprintf(stderr, "Error flushing profile file\n");
        fclose(f_profile);
        exit(EXIT_FAILURE);
    }

    fclose(f_profile);
}

void output_fg(int istep) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    char fg_filename[512];
    snprintf(fg_filename, sizeof(fg_filename), "%s/fg%09d.dat", dirname, istep);

    CHECK_CUDA_ERROR(cudaMemcpy(h_f, d_f, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_g, d_g, NPOP * LXYZ * sizeof(double), cudaMemcpyDeviceToHost));

    FILE *f_fg = fopen(fg_filename, "wb");
    if (f_fg == NULL) {
        fprintf(stderr, "Cannot open file: %s\n", fg_filename);
        exit(EXIT_FAILURE);
    }

    size_t written_elements;
    written_elements = fwrite(h_f, sizeof(double), NPOP * LXYZ, f_fg);
    if (written_elements != NPOP * LXYZ) {
        fprintf(stderr, "Error writing h_f data: wrote %zu of %d elements\n", 
                written_elements, NPOP * LXYZ);
        fclose(f_fg);
        exit(EXIT_FAILURE);
    }
    
    written_elements = fwrite(h_g, sizeof(double), NPOP * LXYZ, f_fg);
    if (written_elements != NPOP * LXYZ) {
        fprintf(stderr, "Error writing h_g data: wrote %zu of %d elements\n", 
                written_elements, NPOP * LXYZ);
        fclose(f_fg);
        exit(EXIT_FAILURE);
    }
    
    if (fflush(f_fg) != 0) {
        fprintf(stderr, "Error flushing fg file\n");
        fclose(f_fg);
        exit(EXIT_FAILURE);
    }
    
    fclose(f_fg);
}


void output_nu_parallel(int istep, int num_gpus) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        #ifdef _WIN32
            _mkdir(dirname);
        #else
            mkdir(dirname, 0700);
        #endif
        printf("Created directory: %s\n", dirname);
    }

    char filename[512]; 
    if (rayl == 1e5) {
        snprintf(filename, sizeof(filename), "%s/Nu_1e50.txt", dirname);
    } else if (rayl == 5e5) {
        snprintf(filename, sizeof(filename), "%s/Nu_5e50.txt", dirname);
    } else if (rayl == 1e6) {
        snprintf(filename, sizeof(filename), "%s/Nu_1e60.txt", dirname);
    } else if (rayl == 5e6) {
        snprintf(filename, sizeof(filename), "%s/Nu_5e60.txt", dirname);
    } else if (rayl == 1e7) {
        snprintf(filename, sizeof(filename), "%s/Nu_1e70.txt", dirname);
    } else if (rayl == 1e8) {
        snprintf(filename, sizeof(filename), "%s/Nu_1e80.txt", dirname);
    } else {
        snprintf(filename, sizeof(filename), "%s/Nu_%.2e.txt", dirname, rayl);
    }

    FILE *f_nu = fopen(filename, "a");
    if (f_nu == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    double global_uyT_sum = 0.0;
    long long global_count = 0;

    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        GPU_Domain *domain = gpu_manager->domains[gpu_id];
        CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
        
        size_t local_field_size = domain->local_size * sizeof(double);
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_uy_local, domain->d_uy_local, local_field_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(domain->h_phi_local, domain->d_phi_local, local_field_size, cudaMemcpyDeviceToHost));
        
        for (int iz = 0; iz < LZ; iz++) {
            for (int iy_local = HALO_WIDTH; iy_local < domain->LY_local_with_halo - HALO_WIDTH; iy_local++) {
                for (int ix = 0; ix < LX; ix++) {
                    int idx_local = get_local_index(ix, iy_local, iz, domain->LY_local_with_halo);
                    global_uyT_sum += domain->h_uy_local[idx_local] * domain->h_phi_local[idx_local];
                    global_count++;
                }
            }
        }
    }

    double uyT_avg = global_uyT_sum / global_count;
    double Nu_bulk = 1.0 + uyT_avg * LY / (diff * (tHot - tCold));

    double dTdy_top = 0.0, dTdy_bottom = 0.0;
    int count_top = 0, count_bottom = 0;
    
    if (num_gpus > 0) {
        GPU_Domain *bottom_domain = gpu_manager->domains[0];
        CHECK_CUDA_ERROR(cudaSetDevice(0));
        
        size_t local_field_size = bottom_domain->local_size * sizeof(double);
        CHECK_CUDA_ERROR(cudaMemcpy(bottom_domain->h_phi_local, bottom_domain->d_phi_local, local_field_size, cudaMemcpyDeviceToHost));
        
        for (int ix = 0; ix < LX; ix++) {
            for (int iz = 0; iz < LZ; iz++) {
                int iy_local = HALO_WIDTH; // 底部物理边界
                int idx_local = get_local_index(ix, iy_local, iz, bottom_domain->LY_local_with_halo);
                int idx_p1 = get_local_index(ix, iy_local + 1, iz, bottom_domain->LY_local_with_halo);
                int idx_p2 = get_local_index(ix, iy_local + 2, iz, bottom_domain->LY_local_with_halo);
                
                double Ty = (-3.0 * bottom_domain->h_phi_local[idx_local] + 
                            4.0 * bottom_domain->h_phi_local[idx_p1] - 
                            bottom_domain->h_phi_local[idx_p2]) / 2.0;
                
                dTdy_bottom += Ty;
                count_bottom++;
            }
        }
    }
    
    if (num_gpus > 0) {
        GPU_Domain *top_domain = gpu_manager->domains[num_gpus - 1];
        CHECK_CUDA_ERROR(cudaSetDevice(num_gpus - 1));
        
        size_t local_field_size = top_domain->local_size * sizeof(double);
        CHECK_CUDA_ERROR(cudaMemcpy(top_domain->h_phi_local, top_domain->d_phi_local, local_field_size, cudaMemcpyDeviceToHost));
        
        for (int ix = 0; ix < LX; ix++) {
            for (int iz = 0; iz < LZ; iz++) {
                int iy_local = top_domain->LY_local_with_halo - HALO_WIDTH - 1; // 顶部物理边界
                int idx_local = get_local_index(ix, iy_local, iz, top_domain->LY_local_with_halo);
                int idx_m1 = get_local_index(ix, iy_local - 1, iz, top_domain->LY_local_with_halo);
                int idx_m2 = get_local_index(ix, iy_local - 2, iz, top_domain->LY_local_with_halo);
                
                double Ty = (3.0 * top_domain->h_phi_local[idx_local] - 
                            4.0 * top_domain->h_phi_local[idx_m1] + 
                            top_domain->h_phi_local[idx_m2]) / 2.0;
                
                dTdy_top += Ty;
                count_top++;
            }
        }
    }
    
    dTdy_top /= count_top;
    dTdy_bottom /= count_bottom;
    
    double Nu_top = -dTdy_top * LY / (tHot - tCold);
    double Nu_bottom = -dTdy_bottom * LY / (tHot - tCold);

    int write_status = fprintf(f_nu, "%15.6E%15.6E%15.6E%15.6E%15.6E\n", 
                             (double)istep, 0.0, Nu_bulk, Nu_top, Nu_bottom);
    if (write_status < 0) {
        fprintf(stderr, "Error writing to Nu file\n");
        fclose(f_nu);
        exit(EXIT_FAILURE);
    }
    
    if (fflush(f_nu) != 0) {
        fprintf(stderr, "Error flushing Nu file\n");
        fclose(f_nu);
        exit(EXIT_FAILURE);
    }
    
    fclose(f_nu);
}

void output_profile_parallel(int istep, int num_gpus) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        #ifdef _WIN32
            _mkdir(dirname);
        #else
            mkdir(dirname, 0700);
        #endif
        printf("Created directory: %s\n", dirname);
    }

    char filename[512]; 
    if (rayl == 1e5) {
        snprintf(filename, sizeof(filename), "%s/Profile_1e50.txt", dirname);
    } else if (rayl == 5e5) {
        snprintf(filename, sizeof(filename), "%s/Profile_5e50.txt", dirname);
    } else if (rayl == 1e6) {
        snprintf(filename, sizeof(filename), "%s/Profile_1e60.txt", dirname);
    } else if (rayl == 5e6) {
        snprintf(filename, sizeof(filename), "%s/Profile_5e60.txt", dirname);
    } else if (rayl == 1e7) {
        snprintf(filename, sizeof(filename), "%s/Profile_1e70.txt", dirname);
    } else if (rayl == 1e8) {
        snprintf(filename, sizeof(filename), "%s/Profile_1e80.txt", dirname);
    } else {
        snprintf(filename, sizeof(filename), "%s/Profile_%.2e.txt", dirname, rayl);
    }

    FILE *f_profile = fopen(filename, "a");
    if (f_profile == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    int write_status = fprintf(f_profile, "%d\n", istep);
    if (write_status < 0) {
        fprintf(stderr, "Error writing step number to profile file\n");
        fclose(f_profile);
        exit(EXIT_FAILURE);
    }

    for (int iy_global = 0; iy_global < LY; iy_global++) {
        double sum_ux = 0.0, sum_uy = 0.0, sum_uz = 0.0, sum_phi = 0.0;
        double sum_phi_sq = 0.0;
        int count = 0;

        int responsible_gpu = iy_global / GET_LY_LOCAL(num_gpus);
        if (responsible_gpu >= num_gpus) responsible_gpu = num_gpus - 1;
        
        GPU_Domain *domain = gpu_manager->domains[responsible_gpu];
        
        int iy_local = iy_global - domain->y_start_global + HALO_WIDTH;
        
        if (iy_local >= HALO_WIDTH && iy_local < domain->LY_local_with_halo - HALO_WIDTH) {
            CHECK_CUDA_ERROR(cudaSetDevice(responsible_gpu));
            size_t local_field_size = domain->local_size * sizeof(double);
            CHECK_CUDA_ERROR(cudaMemcpy(domain->h_ux_local, domain->d_ux_local, local_field_size, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(domain->h_uy_local, domain->d_uy_local, local_field_size, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(domain->h_uz_local, domain->d_uz_local, local_field_size, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(domain->h_phi_local, domain->d_phi_local, local_field_size, cudaMemcpyDeviceToHost));
            
            for (int ix = 0; ix < LX; ix++) {
                for (int iz = 0; iz < LZ; iz++) {
                    int idx_local = get_local_index(ix, iy_local, iz, domain->LY_local_with_halo);

                    sum_ux    += domain->h_ux_local[idx_local];
                    sum_uy    += domain->h_uy_local[idx_local];
                    sum_uz    += domain->h_uz_local[idx_local];
                    sum_phi   += domain->h_phi_local[idx_local];
                    sum_phi_sq += domain->h_phi_local[idx_local] * domain->h_phi_local[idx_local];
                    count++;
                }
            }
        }

        double avg_ux  = sum_ux  / count;
        double avg_uy  = sum_uy  / count;
        double avg_uz  = sum_uz  / count;
        double avg_phi = sum_phi / count;
        double phi_rms = sqrt(sum_phi_sq / count - avg_phi * avg_phi);

        write_status = fprintf(f_profile, "%14.6e %14.6e %14.6e %14.6e %14.6e\n",
                             avg_ux, avg_uy, avg_uz, avg_phi, phi_rms);
        if (write_status < 0) {
            fprintf(stderr, "Error writing profile data\n");
            fclose(f_profile);
            exit(EXIT_FAILURE);
        }
    }

    if (fflush(f_profile) != 0) {
        fprintf(stderr, "Error flushing profile file\n");
        fclose(f_profile);
        exit(EXIT_FAILURE);
    }

    fclose(f_profile);
}

void output_fg_parallel(int istep, int num_gpus) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), "Ra%.1ePr%.2f", rayl, prand);

    char fg_filename[512];
    snprintf(fg_filename, sizeof(fg_filename), "%s/fg%09d.dat", dirname, istep);

    FILE *f_fg = fopen(fg_filename, "wb");
    if (f_fg == NULL) {
        fprintf(stderr, "Cannot open file: %s\n", fg_filename);
        exit(EXIT_FAILURE);
    }

    for (int is_g = 0; is_g < 2; is_g++) {
        for (int ip = 0; ip < NPOP; ip++) {
            for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
                GPU_Domain *domain = gpu_manager->domains[gpu_id];
                CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));

                size_t local_dist_size = NPOP * domain->local_size * sizeof(double);
                double *h_dist_local = (double*)malloc(local_dist_size);
                if (!h_dist_local) {
                    fprintf(stderr, "Failed to allocate memory for distribution function output\n");
                    exit(EXIT_FAILURE);
                }
                
                if (is_g == 0) {
                    CHECK_CUDA_ERROR(cudaMemcpy(h_dist_local, domain->d_f_local, local_dist_size, cudaMemcpyDeviceToHost));
                } else {
                    CHECK_CUDA_ERROR(cudaMemcpy(h_dist_local, domain->d_g_local, local_dist_size, cudaMemcpyDeviceToHost));
                }
                
                for (int iz = 0; iz < LZ; iz++) {
                    for (int iy_local = HALO_WIDTH; iy_local < domain->LY_local_with_halo - HALO_WIDTH; iy_local++) {
                        for (int ix = 0; ix < LX; ix++) {
                            int idx_local = get_local_index(ix, iy_local, iz, domain->LY_local_with_halo);
                            double value = h_dist_local[ip * domain->local_size + idx_local];
                            
                            size_t written = fwrite(&value, sizeof(double), 1, f_fg);
                            if (written != 1) {
                                fprintf(stderr, "Error writing distribution function data\n");
                                free(h_dist_local);
                                fclose(f_fg);
                                exit(EXIT_FAILURE);
                            }
                        }
                    }
                }
                
                free(h_dist_local);
            }
        }
    }

    if (fflush(f_fg) != 0) {
        fprintf(stderr, "Error flushing fg file\n");
        fclose(f_fg);
        exit(EXIT_FAILURE);
    }
    
    fclose(f_fg);
}

