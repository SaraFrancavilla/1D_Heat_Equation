#include "all_implementation.h"

void compareResults(int Nx, int Nt, float central_heat, double s, int rank, int size, int index){

    if (rank == 0) {
        create_clean_file("Results_comparison.txt");
    } 
    
    if (rank == 0) {

        solveSeqFullMatrix(Nx, Nt, central_heat, s, index);

        solveSeqHalfMatrix(Nx, Nt, central_heat, s, index);

        solveSeqSwapVec(Nx, Nt, central_heat, s, index);

        solveOMPHalfMatrixUNROLLL(Nx, Nt, central_heat, s, index);

        solveOMPHalfMatrixCOLLAPSE(Nx, Nt, central_heat, s, index);

        // solveOMPHalfMatrixBARRIER(Nx, Nt, central_heat, s, index);

        solveOMPHalfMatrixALIGNED(Nx, Nt, central_heat, s, index);

        solveOMPSwapVecALIGNED(Nx, Nt, central_heat, s, index);
    }
    
    int color;
    if (rank < matrix_configs[5]) {
        color = 1;  // Processi attivi
    } else {
        color = MPI_UNDEFINED;  // Processi inattivi esclusi
    }

    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &active_comm);

    if (color == 1) {

        int active_rank, active_size;
        MPI_Comm_rank(active_comm, &active_rank);
        MPI_Comm_size(active_comm, &active_size);

        double time = 0.0;

        solveMPIHalfMatrixBROADCAST(Nx, Nt, central_heat, s, rank, size, index, active_comm, &time);

        solveMPIHalfMatrixBMATRIX(Nx, Nt, central_heat, s, rank, size, index, active_comm, &time);

        solveMPIHalfMatrixSMATRIX(Nx, Nt, central_heat, s, rank, size, index, active_comm, &time);
        
        solveMPIHalfMatrixSENDREC(Nx, Nt, central_heat, s, rank, size, index, active_comm, &time);

        solveMPIHalfMatrixISENDIRECV(Nx, Nt, central_heat, s, rank, size, index, active_comm, &time);

        MPI_Comm_free(&active_comm);  // Libera il comunicatore
    }   

}
void executionTimes(int rank, int index){

    double start_time, end_time, elapsed_time;
    double seq_times[3], speedup[3], efficiency[3];

    FILE *file = NULL;

    if (rank == 0) {
        create_clean_file("Execution_times.txt");

        file = fopen("Execution_times.txt", "a");
        if (file == NULL) {
            perror("Error opening file for writing results");
            return;
        }

        fprintf(file, "\nS -> Speedup/Strong scaling\nE -> Efficiency \n");
    }


    for (int i = 1; i < MATRIX_LINES; i++){

        if (rank == 0) {

            fprintf(file, "\n\nMATRIX %.0f x %.0f \n", matrix_configs[i * MATRIX_COLUMNS], matrix_configs[i * MATRIX_COLUMNS + 1]);

            //SEQUENTIAL
            fprintf(file, "\nSequential\n========================\n");
            //Sequential Full Matrix
            start_time = MPI_Wtime();
            // solveSeqFullMatrix(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], index);
            solveSeqFullMatrix(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS+1], matrix_configs[(i)*MATRIX_COLUMNS +2], matrix_configs[(i)*MATRIX_COLUMNS +3], index);
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;
            fprintf(file, "Full Matrix: %.9f\n", elapsed_time);
            seq_times[0] = elapsed_time;

            //Sequential Half Matrix
            start_time = MPI_Wtime();
            // solveSeqHalfMatrix(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], index);
            solveSeqHalfMatrix(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS+1], matrix_configs[(i)*MATRIX_COLUMNS +2], matrix_configs[(i)*MATRIX_COLUMNS +3], index);
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;
            fprintf(file, "Half Matrix: %.9f\n", elapsed_time);
            seq_times[1] = elapsed_time;

            //Sequential Swap Vector
            start_time = MPI_Wtime();
            // solveSeqSwapVec(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], index);
            solveSeqSwapVec(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS+1], matrix_configs[(i)*MATRIX_COLUMNS +2], matrix_configs[(i)*MATRIX_COLUMNS +3], index);
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;
            fprintf(file, "Swap Vector: %.9f\n", elapsed_time);
            seq_times[2] = elapsed_time;


            //OPENMP
            fprintf(file, "\n OpenMP                                                                 Full         Half         Swap");
            fprintf(file,"\n======================================                          ===========================================\n");
            

            //setting num of threads depending on matrix dimensions
            omp_set_num_threads(matrix_configs[(i + 5) * MATRIX_COLUMNS]);

            fprintf(file, "Half Matrix Unroll:          ");
            start_time = MPI_Wtime();
            // solveOMPHalfMatrixUNROLLL(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], index);
            solveOMPHalfMatrixUNROLLL(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS+1], matrix_configs[(i)*MATRIX_COLUMNS +2], matrix_configs[(i)*MATRIX_COLUMNS +3], index);
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;
            for (int t = 0; t < 3; t++){
                speedup[t] = (seq_times[t]/elapsed_time);
                efficiency[t] = (speedup[t]/matrix_configs[(i) * MATRIX_COLUMNS + 4])*100;
            }
            fprintf(file, "  %.9f                       S:  %.9f  %.9f  %.9f\n", elapsed_time, speedup[0], speedup[1], speedup[2]);
            fprintf(file, "                                                                 ");
            fprintf(file, "E:  %.9f  %.9f  %.9f\n", efficiency[0], efficiency[1], efficiency[2]);


            fprintf(file, "\nHalf Matrix Collapse:        ");

            start_time = MPI_Wtime();
            // solveOMPHalfMatrixCOLLAPSE(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], index);
            solveOMPHalfMatrixCOLLAPSE(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS+1], matrix_configs[(i)*MATRIX_COLUMNS +2], matrix_configs[(i)*MATRIX_COLUMNS +3], index);
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;
            for (int t = 0; t < 3; t++){
                speedup[t] = (seq_times[t]/elapsed_time);
                efficiency[t] = (speedup[t]/matrix_configs[(i) * MATRIX_COLUMNS + 4])*100;
            }
            fprintf(file, "  %.9f                       S:  %.9f  %.9f  %.9f\n", elapsed_time, speedup[0], speedup[1], speedup[2]);
            fprintf(file, "                                                                 ");
            fprintf(file, "E:  %.9f  %.9f  %.9f\n", efficiency[0], efficiency[1], efficiency[2]);
            
            // fprintf(file, "\nHalf Matrix Barrier:         ");
            //     start_time = MPI_Wtime();
            //     solveOMPHalfMatrixBARRIER(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], index);
            //     end_time = MPI_Wtime();
            //     elapsed_time = end_time - start_time;
            // for (int i = 0; i < 3; i++){
            //     speedup[i] = (seq_times[i]/elapsed_time);
            //     efficiency[i] = (speedup[i]/omp_get_num_threads());
            // }
            // fprintf(file, "  %.9f                       S:  %.9f  %.9f  %.9f\n", elapsed_time, speedup[0], speedup[1], speedup[2]);
            // fprintf(file, "                                                                 ");
            // fprintf(file, "E:  %.9f  %.9f  %.9f\n", efficiency[0], efficiency[1], efficiency[2]);

            fprintf(file, "\nHalf Matrix Aligned:         ");
            start_time = MPI_Wtime();
            // solveOMPHalfMatrixALIGNED(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], index);
            solveOMPHalfMatrixALIGNED(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS+1], matrix_configs[(i)*MATRIX_COLUMNS +2], matrix_configs[(i)*MATRIX_COLUMNS +3], index);
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;
            for (int t = 0; t < 3; t++){
                speedup[t] = (seq_times[t]/elapsed_time);
                efficiency[t] = (speedup[t]/matrix_configs[(i) * MATRIX_COLUMNS + 4])*100;
            }
            fprintf(file, "  %.9f                       S:  %.9f  %.9f  %.9f\n", elapsed_time, speedup[0], speedup[1], speedup[2]);
            fprintf(file, "                                                                 ");
            fprintf(file, "E:  %.9f  %.9f  %.9f\n", efficiency[0], efficiency[1], efficiency[2]);

            fprintf(file, "\nSwap Vector Aligned:         ");
            start_time = MPI_Wtime();
            // solveOMPSwapVecALIGNED(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], index);
            solveOMPSwapVecALIGNED(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS+1], matrix_configs[(i)*MATRIX_COLUMNS +2], matrix_configs[(i)*MATRIX_COLUMNS +3], index);
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;
            for (int t = 0; t < 3; t++){
                speedup[t] = (seq_times[t]/elapsed_time);
                efficiency[t] = (speedup[t]/matrix_configs[(i) * MATRIX_COLUMNS + 4])*100;
            }
            fprintf(file, "  %.9f                       S:  %.9f  %.9f  %.9f\n", elapsed_time, speedup[0], speedup[1], speedup[2]);
            fprintf(file, "                                                                 ");
            fprintf(file, "E:  %.9f  %.9f  %.9f\n", efficiency[0], efficiency[1], efficiency[2]);

            fprintf(file, "\n\nMPI                                                                 Full         Half         Swap");
            fprintf(file, "\n======================================                          ===========================================\n");
            fprintf(file, "Half Matrix Broadcast:       ");
            fflush(file);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        int color;
        // if (rank < matrix_configs[(i + 4) * MATRIX_COLUMNS]) {
        if (rank < matrix_configs[(i - 1) * MATRIX_COLUMNS + 5 ]) {
            color = 1;  // Processi attivi
        } else {
            color = MPI_UNDEFINED;  // Processi inattivi esclusi
        }
    
        MPI_Comm active_comm;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &active_comm);
    
        if (color == 1) {
            int active_rank, active_size;
            MPI_Comm_rank(active_comm, &active_rank);
            MPI_Comm_size(active_comm, &active_size);
            double comm_time = 0.0;
            float perc_time = 0.0;
        
            start_time = MPI_Wtime();
            // solveMPIHalfMatrixBROADCAST(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
            solveMPIHalfMatrixBROADCAST(matrix_configs[(i) * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS +1], matrix_configs[(i)*MATRIX_COLUMNS] +2, matrix_configs[(i)*MATRIX_COLUMNS +3], active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;
        
            if (active_rank == 0 && rank == 0) {
                elapsed_time = end_time - start_time;
                perc_time = (comm_time/elapsed_time) * 100;
                for (int t = 0; t < 3; t++){
                    speedup[t] = (seq_times[t]/elapsed_time);
                    efficiency[t] = (speedup[t]/matrix_configs[(i) * MATRIX_COLUMNS + 5])*100;
                }
                fprintf(file, "  %.9f                       S:  %.9f  %.9f  %.9f\n", elapsed_time, speedup[0], speedup[1], speedup[2]);
                //fprintf(file, "                                                                 ");
                fprintf(file, "       Comm_time: %.9f  %.8f%% of time_tot           ", comm_time, perc_time);
                fprintf(file, "E:  %.9f  %.9f  %.9f\n", efficiency[0], efficiency[1], efficiency[2]);
                fprintf(file, "\nHalf Matrix Broadcast Matrix:");
            }

            comm_time = 0.0;
            start_time = MPI_Wtime();
            // solveMPIHalfMatrixBMATRIX(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
            solveMPIHalfMatrixBMATRIX(matrix_configs[(i) * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS +1], matrix_configs[(i)*MATRIX_COLUMNS] +2, matrix_configs[(i)*MATRIX_COLUMNS +3], active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;

            if (active_rank == 0 && rank == 0) {
                elapsed_time = end_time - start_time;
                perc_time = (comm_time/elapsed_time) * 100;
                for (int t = 0; t < 3; t++){
                    speedup[t] = (seq_times[t]/elapsed_time);
                    efficiency[t] = (speedup[t]/matrix_configs[(i) * MATRIX_COLUMNS + 5])*100;
                }
                fprintf(file, "  %.9f                       S:  %.9f  %.9f  %.9f\n", elapsed_time, speedup[0], speedup[1], speedup[2]);
                //fprintf(file, "                                                                 ");
                fprintf(file, "       Comm_time: %.9f  %.8f%% of time_tot           ", comm_time, perc_time);
                fprintf(file, "E:  %.9f  %.9f  %.9f\n", efficiency[0], efficiency[1], efficiency[2]);
                fprintf(file, "\nHalf Matrix Scatter Matrix:  ");
            }

            comm_time = 0.0;
            start_time = MPI_Wtime();
            // solveMPIHalfMatrixSMATRIX(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
            solveMPIHalfMatrixSMATRIX(matrix_configs[(i) * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS +1], matrix_configs[(i)*MATRIX_COLUMNS] +2, matrix_configs[(i)*MATRIX_COLUMNS +3], active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;

            if (active_rank == 0 && rank == 0) {
                elapsed_time = end_time - start_time;
                perc_time = (comm_time/elapsed_time) * 100;
                for (int t = 0; t < 3; t++){
                    speedup[t] = (seq_times[t]/elapsed_time);
                    efficiency[t] = (speedup[t]/matrix_configs[(i) * MATRIX_COLUMNS + 5])*100;
                }
                fprintf(file, "  %.9f                       S:  %.9f  %.9f  %.9f\n", elapsed_time, speedup[0], speedup[1], speedup[2]);
                //fprintf(file, "                                                                 ");
                fprintf(file, "       Comm_time: %.9f  %.8f%% of time_tot           ", comm_time, perc_time);
                fprintf(file, "E:  %.9f  %.9f  %.9f\n", efficiency[0], efficiency[1], efficiency[2]);
                fprintf(file, "\nHalf Matrix Send Recv:       ");
            }

            comm_time = 0.0;
            start_time = MPI_Wtime();
            // solveMPIHalfMatrixSENDREC(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
            solveMPIHalfMatrixSENDREC(matrix_configs[(i) * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS +1], matrix_configs[(i)*MATRIX_COLUMNS] +2, matrix_configs[(i)*MATRIX_COLUMNS +3], active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;

            if (active_rank == 0 && rank == 0) {
                elapsed_time = end_time - start_time;
                perc_time = (comm_time/elapsed_time) * 100;
                for (int t = 0; t < 3; t++){
                    speedup[t] = (seq_times[t]/elapsed_time);
                    efficiency[t] = (speedup[t]/matrix_configs[(i) * MATRIX_COLUMNS + 5])*100;
                }
                fprintf(file, "  %.9f                       S:  %.9f  %.9f  %.9f\n", elapsed_time, speedup[0], speedup[1], speedup[2]);
                //fprintf(file, "                                                                 ");
                fprintf(file, "       Comm_time: %.9f  %.8f%% of time_tot           ", comm_time, perc_time);
                fprintf(file, "E:  %.9f  %.9f  %.9f\n", efficiency[0], efficiency[1], efficiency[2]);
                fprintf(file, "\nHalf Matrix ISend IRecv:     ");
            }

            comm_time = 0.0;
            start_time = MPI_Wtime();
            // solveMPIHalfMatrixISENDIRECV(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i+1)*MATRIX_COLUMNS], matrix_configs[(i+2)*MATRIX_COLUMNS], matrix_configs[(i+3)*MATRIX_COLUMNS], active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
            solveMPIHalfMatrixISENDIRECV(matrix_configs[(i) * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS +1], matrix_configs[(i)*MATRIX_COLUMNS] +2, matrix_configs[(i)*MATRIX_COLUMNS +3], active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
            end_time = MPI_Wtime();
            elapsed_time = end_time - start_time;
        
            if (active_rank == 0 && rank == 0) {
                elapsed_time = end_time - start_time;
                perc_time = (comm_time/elapsed_time) * 100;
                for (int t = 0; t < 3; t++){
                    speedup[t] = (seq_times[t]/elapsed_time);
                    efficiency[t] = (speedup[t]/matrix_configs[(i) * MATRIX_COLUMNS + 5])*100;
                }
                fprintf(file, "  %.9f                       S:  %.9f  %.9f  %.9f\n", elapsed_time, speedup[0], speedup[1], speedup[2]);
                //fprintf(file, "                                                                 ");
                fprintf(file, "       Comm_time: %.9f  %.8f%% of time_tot           ", comm_time, perc_time);
                fprintf(file, "E:  %.9f  %.9f  %.9f\n", efficiency[0], efficiency[1], efficiency[2]);
            }


            MPI_Comm_free(&active_comm);  // Libera il comunicatore
        }
    }

    if (rank == 0 && file != NULL) {
        fclose(file);
    }

}

void searchingOptimalConfiguration(int rank, int size, int index){

    double start_time, end_time, elapsed_time, sequential_time, swap_time;
    double *scalability = malloc(6 * sizeof(double));

    FILE *file = NULL;

    if (rank == 0) {
        create_clean_file("Optimal_configurations_study.txt");

        file = fopen("Optimal_configurations_study.txt", "a");
        if (file == NULL) {
            perror("Error opening file for writing results");
            return;
        }

        fprintf(file, "\nT -> Time of execution\n");
        fprintf(file, "\nS -> Scalability/Weak scaling\n");
    }

    double best_time[2];

    for (int i = 1; i < MATRIX_LINES; i++){

        if (rank == 0) {

            fprintf(file, "\n\nMATRIX %.0f x %.0f \n", matrix_configs[i * MATRIX_COLUMNS], matrix_configs[i * MATRIX_COLUMNS + 1]);

            //Computing sequential time
            start_time = MPI_Wtime();
            solveSeqHalfMatrix(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS+1], matrix_configs[(i)*MATRIX_COLUMNS +2], matrix_configs[(i)*MATRIX_COLUMNS +3], index);
            end_time = MPI_Wtime();
            sequential_time = end_time - start_time;

            start_time = MPI_Wtime();
            solveSeqSwapVec(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[(i)*MATRIX_COLUMNS+1], matrix_configs[(i)*MATRIX_COLUMNS +2], matrix_configs[(i)*MATRIX_COLUMNS +3], index);
            end_time = MPI_Wtime();
            swap_time = end_time - start_time;

            //OPENMP
            fprintf(file, "\nOpenMP\n======================================\n");

            fprintf(file, "Half Matrix Unroll:          ");

            for (int t= 0; t < THREAD_SIZE; t++){
                omp_set_num_threads(num_threads[t]);
                start_time = MPI_Wtime();
                solveOMPHalfMatrixUNROLLL(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[i * MATRIX_COLUMNS + 1], matrix_configs[i * MATRIX_COLUMNS + 2], matrix_configs[i * MATRIX_COLUMNS + 3], index);
                end_time = MPI_Wtime();
                elapsed_time = end_time - start_time;
                scalability[t] = sequential_time / elapsed_time;
                fprintf(file, "T: %.9f  ", elapsed_time);
                if (t == 0){
                    best_time[0] = elapsed_time;
                    best_time[1] = pow(2, t+1);
                } else if (elapsed_time < best_time[0]){
                    best_time[0] = elapsed_time;
                    best_time[1] = pow(2,t+1);
                }
            }
            fprintf(file, "    Best Time: %.9f (with %.0f threads)\n", best_time[0], best_time[1]);
            fprintf(file, "                             ");
            for (int t= 0; t < THREAD_SIZE; t++){
                fprintf(file, "S: %.9f  ", scalability[t]);
            }

            fprintf(file, "\n\nHalf Matrix Collapse:        ");

            for (int t= 0; t < THREAD_SIZE; t++){
                omp_set_num_threads(num_threads[t]);
                start_time = MPI_Wtime();
                solveOMPHalfMatrixCOLLAPSE(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[i * MATRIX_COLUMNS + 1], matrix_configs[i * MATRIX_COLUMNS + 2], matrix_configs[i * MATRIX_COLUMNS + 3], index);
                end_time = MPI_Wtime();
                elapsed_time = end_time - start_time;
                scalability[t] = sequential_time / elapsed_time;
                fprintf(file, "T: %.9f  ", elapsed_time);
                if (t == 0){
                    best_time[0] = elapsed_time;
                    best_time[1] = pow(2,t +1);
                } else if (elapsed_time < best_time[0]){
                    best_time[0] = elapsed_time;
                    best_time[1] = pow(2,t+1);
                }
            }
            fprintf(file, "    Best Time: %.9f (with %.0f threads)\n", best_time[0], best_time[1]);
            fprintf(file, "                             ");
            for (int t= 0; t < THREAD_SIZE; t++){
                fprintf(file, "S: %.9f  ", scalability[t]);
            }

            fprintf(file, "\n\nHalf Matrix Aligned:         ");
            for (int t= 0; t < THREAD_SIZE; t++){
                omp_set_num_threads(num_threads[t]);
                start_time = MPI_Wtime();
                solveOMPHalfMatrixALIGNED(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[i * MATRIX_COLUMNS + 1], matrix_configs[i * MATRIX_COLUMNS + 2], matrix_configs[i * MATRIX_COLUMNS + 3], index);
                end_time = MPI_Wtime();
                elapsed_time = end_time - start_time;
                scalability[t] = sequential_time / elapsed_time;
                fprintf(file, "T: %.9f  ", elapsed_time);
                if (t == 0){
                    best_time[0] = elapsed_time;
                    best_time[1] = pow(2,t+1);
                } else if (elapsed_time < best_time[0]){
                    best_time[0] = elapsed_time;
                    best_time[1] = pow(2,t+1);
                }
            }
            fprintf(file, "    Best Time: %.9f (with %.0f threads)\n", best_time[0], best_time[1]);
            fprintf(file, "                             ");
            for (int t= 0; t < THREAD_SIZE; t++){
                fprintf(file, "S: %.9f  ", scalability[t]);
            }

            fprintf(file, "\n\nSwap Vector Aligned:         ");

            for (int t= 0; t < THREAD_SIZE; t++){
                omp_set_num_threads(num_threads[t]);
                start_time = MPI_Wtime();
                solveOMPSwapVecALIGNED(matrix_configs[i * MATRIX_COLUMNS], matrix_configs[i * MATRIX_COLUMNS + 1], matrix_configs[i * MATRIX_COLUMNS + 2], matrix_configs[i * MATRIX_COLUMNS + 3], index);
                end_time = MPI_Wtime();
                elapsed_time = end_time - start_time;
                scalability[t] = swap_time / elapsed_time;
                fprintf(file, "T: %.9f  ", elapsed_time);
                if (t == 0){
                    best_time[0] = elapsed_time;
                    best_time[1] = pow(2,t+1);
                } else if (elapsed_time < best_time[0]){
                    best_time[0] = elapsed_time;
                    best_time[1] = pow(2,t+1);
                }
            }
            fprintf(file, "    Best Time: %.9f (with %.0f threads)\n", best_time[0], best_time[1]);
            fprintf(file, "                             ");
            for (int t= 0; t < THREAD_SIZE; t++){
                fprintf(file, "S: %.9f  ", scalability[t]);
            }

        }
        if (rank == 0){
            fprintf(file, "\n\n\nMPI \n=========================================\n");
            fprintf(file, "Half Matrix Broadcast:         ");
        }
        
        for (int t= 0; t < PROCESS_SIZE; t++){

            MPI_Barrier(MPI_COMM_WORLD);
            int color;
            if (rank < num_processes[t]) {
                color = 1;  // Processi attivi
            } else {
                color = MPI_UNDEFINED;  // Processi inattivi esclusi
            }
        
            MPI_Comm active_comm;
            MPI_Comm_split(MPI_COMM_WORLD, color, rank, &active_comm);
        
            if (color == 1) {
                int active_rank, active_size;
                MPI_Comm_rank(active_comm, &active_rank);
                MPI_Comm_size(active_comm, &active_size);

                double comm_time = 0.0;
            
                start_time = MPI_Wtime();
                solveMPIHalfMatrixBROADCAST(matrix_configs[i * MATRIX_COLUMNS], 
                                            matrix_configs[i * MATRIX_COLUMNS + 1], 
                                            matrix_configs[i * MATRIX_COLUMNS + 2], 
                                            matrix_configs[i * MATRIX_COLUMNS + 3], 
                                            active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
                end_time = MPI_Wtime();
                elapsed_time = end_time - start_time;
                scalability[t] = sequential_time / elapsed_time;
            
                if (active_rank == 0 && rank == 0) {
                    fprintf(file, "T: %.9f  ", elapsed_time);
                    if (t == 0){
                        best_time[0] = elapsed_time;
                        best_time[1] = pow(2,t+1);
                    } else if (elapsed_time < best_time[0]){
                        best_time[0] = elapsed_time;
                        best_time[1] = pow(2,t+1);
                    }
                }

                MPI_Comm_free(&active_comm);  // Libera il comunicatore
            }
        }

        if (rank == 0) {
            
            fprintf(file, "    Best Time: %.9f (with %.0f processes)\n", best_time[0], best_time[1]);
            fprintf(file, "                               ");
            for (int t= 0; t < PROCESS_SIZE; t++){
                fprintf(file, "S: %.9f  ", scalability[t]);
            }
            fprintf(file, "\n\nHalf Matrix Broadcast Matrix:  ");
        }


        for (int t= 0; t < PROCESS_SIZE; t++){

                MPI_Barrier(MPI_COMM_WORLD);
                int color;
                if (rank < num_processes[t]) {
                    color = 1;  // Processi attivi
                } else {
                    color = MPI_UNDEFINED;  // Processi inattivi esclusi
                }
            
                MPI_Comm active_comm;
                MPI_Comm_split(MPI_COMM_WORLD, color, rank, &active_comm);
            
                if (color == 1) {
                    int active_rank, active_size;
                    MPI_Comm_rank(active_comm, &active_rank);
                    MPI_Comm_size(active_comm, &active_size);
                    double comm_time = 0.0;
                
                    start_time = MPI_Wtime();
                    solveMPIHalfMatrixBMATRIX(matrix_configs[i * MATRIX_COLUMNS], 
                                                matrix_configs[i * MATRIX_COLUMNS + 1], 
                                                matrix_configs[i * MATRIX_COLUMNS + 2], 
                                                matrix_configs[i * MATRIX_COLUMNS + 3], 
                                                active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
                    end_time = MPI_Wtime();
                    elapsed_time = end_time - start_time;
                    scalability[t] = sequential_time / elapsed_time;
                
                    if (active_rank == 0 && rank == 0) {
                        fprintf(file, "T: %.9f  ", elapsed_time);
                        if (t == 0){
                            best_time[0] = elapsed_time;
                            best_time[1] = pow(2,t+1);
                        } else if (elapsed_time < best_time[0]){
                            best_time[0] = elapsed_time;
                            best_time[1] = pow(2,t+1);
                        }
                    }

                    MPI_Comm_free(&active_comm);  // Libera il comunicatore
                }
            }

            if (rank == 0) {
                fprintf(file, "    Best Time: %.9f (with %.0f processes)\n", best_time[0], best_time[1]);
                fprintf(file, "                               ");
                for (int t= 0; t < PROCESS_SIZE; t++){
                    fprintf(file, "S: %.9f  ", scalability[t]);
                }
                fprintf(file, "\n\nHalf Matrix Scatter Matrix:    ");
            }
    
    
            for (int t= 0; t < PROCESS_SIZE; t++){
    
                    MPI_Barrier(MPI_COMM_WORLD);
                    int color;
                    if (rank < num_processes[t]) {
                        color = 1;  // Processi attivi
                    } else {
                        color = MPI_UNDEFINED;  // Processi inattivi esclusi
                    }
                
                    MPI_Comm active_comm;
                    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &active_comm);
                
                    if (color == 1) {
                        int active_rank, active_size;
                        MPI_Comm_rank(active_comm, &active_rank);
                        MPI_Comm_size(active_comm, &active_size);
                        double comm_time = 0.0;
                    
                        start_time = MPI_Wtime();
                        solveMPIHalfMatrixSMATRIX(matrix_configs[i * MATRIX_COLUMNS], 
                                                    matrix_configs[i * MATRIX_COLUMNS + 1], 
                                                    matrix_configs[i * MATRIX_COLUMNS + 2], 
                                                    matrix_configs[i * MATRIX_COLUMNS + 3], 
                                                    active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
                        end_time = MPI_Wtime();
                        elapsed_time = end_time - start_time;
                        scalability[t] = sequential_time / elapsed_time;
                
                        if (active_rank == 0 && rank == 0) {
                            fprintf(file, "T: %.9f  ", elapsed_time);
                            if (t == 0){
                                best_time[0] = elapsed_time;
                                best_time[1] = pow(2,t+1);
                            } else if (elapsed_time < best_time[0]){
                                best_time[0] = elapsed_time;
                                best_time[1] = pow(2,t+1);
                            }
                        }
    
                        MPI_Comm_free(&active_comm);  // Libera il comunicatore
                    }
                }

        if (rank == 0){
            fprintf(file, "    Best Time: %.9f (with %.0f processes)\n", best_time[0], best_time[1]);
            fprintf(file, "                               ");
                for (int t= 0; t < PROCESS_SIZE; t++){
                    fprintf(file, "S: %.9f  ", scalability[t]);
                }
            fprintf(file, "\n\nHalf Matrix Send Recv:         ");
        }

        for (int t= 0; t < PROCESS_SIZE; t++){
    
            MPI_Barrier(MPI_COMM_WORLD);
            int color;
            if (rank < num_processes[t]) {
                color = 1;  // Processi attivi
            } else {
                color = MPI_UNDEFINED;  // Processi inattivi esclusi
            }
        
            MPI_Comm active_comm;
            MPI_Comm_split(MPI_COMM_WORLD, color, rank, &active_comm);
        
            if (color == 1) {
                int active_rank, active_size;
                MPI_Comm_rank(active_comm, &active_rank);
                MPI_Comm_size(active_comm, &active_size);
                double comm_time = 0.0;
            
                start_time = MPI_Wtime();
                solveMPIHalfMatrixSENDREC(matrix_configs[i * MATRIX_COLUMNS], 
                                            matrix_configs[i * MATRIX_COLUMNS + 1], 
                                            matrix_configs[i * MATRIX_COLUMNS + 2], 
                                            matrix_configs[i * MATRIX_COLUMNS + 3], 
                                            active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
                end_time = MPI_Wtime();
                elapsed_time = end_time - start_time;
                scalability[t] = sequential_time / elapsed_time;
                
                if (active_rank == 0 && rank == 0) {
                    fprintf(file, "T: %.9f  ", elapsed_time);
                    if (t == 0){
                        best_time[0] = elapsed_time;
                        best_time[1] = pow(2,t+1);
                    } else if (elapsed_time < best_time[0]){
                        best_time[0] = elapsed_time;
                        best_time[1] = pow(2,t+1);
                    }
                }

                MPI_Comm_free(&active_comm);  // Libera il comunicatore
            }
        }
        if (rank == 0){
            
            fprintf(file, "    Best Time: %.9f (with %.0f processes)\n", best_time[0], best_time[1]);
            fprintf(file, "                               ");
            for (int t= 0; t < PROCESS_SIZE; t++){
                fprintf(file, "S: %.9f  ", scalability[t]);
            }
            fprintf(file, "\n\nHalf Matrix ISend IRecv:       ");
        }


        for (int t= 0; t < PROCESS_SIZE; t++){
    
            MPI_Barrier(MPI_COMM_WORLD);
            int color;
            if (rank < num_processes[t]) {
                color = 1;  // Processi attivi
            } else {
                color = MPI_UNDEFINED;  // Processi inattivi esclusi
            }
        
            MPI_Comm active_comm;
            MPI_Comm_split(MPI_COMM_WORLD, color, rank, &active_comm);
        
            if (color == 1) {
                int active_rank, active_size;
                MPI_Comm_rank(active_comm, &active_rank);
                MPI_Comm_size(active_comm, &active_size);
                double comm_time = 0.0;
            
                start_time = MPI_Wtime();
                solveMPIHalfMatrixISENDIRECV(matrix_configs[i * MATRIX_COLUMNS], 
                                            matrix_configs[i * MATRIX_COLUMNS + 1], 
                                            matrix_configs[i * MATRIX_COLUMNS + 2], 
                                            matrix_configs[i * MATRIX_COLUMNS + 3], 
                                            active_rank, active_size, index, active_comm, &comm_time);  // Passa il nuovo comunicatore
                end_time = MPI_Wtime();
                elapsed_time = end_time - start_time;
                scalability[t] = sequential_time / elapsed_time;
                
                if (active_rank == 0 && rank == 0) {
                    fprintf(file, "T: %.9f  ", elapsed_time);
                    if (t == 0){
                        best_time[0] = elapsed_time;
                        best_time[1] = pow(2,t+1);
                    } else if (elapsed_time < best_time[0]){
                        best_time[0] = elapsed_time;
                        best_time[1] = pow(2,t+1);
                    }
                }

                MPI_Comm_free(&active_comm);  // Libera il comunicatore
            }
        }
        if (rank == 0){
            fprintf(file, "    Best Time: %.9f (with %.0f processes)\n", best_time[0], best_time[1]);
            fprintf(file, "                               ");
            for (int t= 0; t < PROCESS_SIZE; t++){
                fprintf(file, "S: %.9f  ", scalability[t]);
            }
        }

    }

    if (rank == 0 && file != NULL) {
        fclose(file);
    }
}


