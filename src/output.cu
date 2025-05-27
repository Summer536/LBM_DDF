#include "rb3d.h"

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