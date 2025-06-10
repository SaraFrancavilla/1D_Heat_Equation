#include "all_implementation.h"

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int index = 1;
    int limit = 4;

    while (index != 0) {

        if (rank == 0) {
            
            printf("Please choose an option:\n");
            printf("Enter the index of your choice: \n");
            printf("0. Exit\n");
            printf("1. Compare results to check proper functioning\n");
            printf("2. Measure the execution time of the different solutions at optimal condition\n");
            printf("   for different spatial and temporal resolutions\n");
            printf("3. Search for the optimal conditions based on the dimension of the matrices\n");
            // Validate input to be an integer
            if (scanf("%d", &index) != 1) {
                printf("Invalid input. Please enter an integer.\n");
                // Clear input buffer
                while (getchar() != '\n');

            } else if (index >= limit || index < 0) {
                // Ensure the integer is within the allowed range
                printf("Invalid choice. Please enter an index between 0 and %d.\n", limit - 1);
            }
            
        }
    
        // Broadcast the index from process 0 to all other processes
        MPI_Bcast(&index, 1, MPI_INT, 0, MPI_COMM_WORLD);

        switch (index)
        {
        case 1:
            if (rank == 0){
                printf("Please check your results in the 'Results_comparison.txt' file\n");
            }
            omp_set_num_threads(matrix_configs[5]);
            compareResults(matrix_configs[0], matrix_configs[1], matrix_configs[2], matrix_configs[3], rank, matrix_configs[4], index);
            
            break;
        
        case 2:
        if (rank == 0){
            printf("Please check your results in the 'Execution_times.txt' file\n");
        }
        executionTimes(rank, index);
        break;
        
        case 3:
            if (rank == 0){
                printf("Please check your results in the 'Optimal_configurations_study.txt' file\n");
            }
            searchingOptimalConfiguration(rank, size, index);

        default:
            break;
        }
    }
    

    MPI_Finalize();

    return 0;


}


