#include "all_implementation.h"


void create_clean_file(const char *filename) {
    // Check if the file exists
    FILE *file = fopen(filename, "r");
    if (file != NULL) {
        // File exists, close it and delete it
        fclose(file);
        if (remove(filename) != 0) {
            perror("Error deleting existing file");
        }
    }
}

void compareResultsFile(int dim, double *u, int id){
    // writing to file
    FILE *file = fopen("Results_comparison.txt", "a");
    if (file == NULL) {
        perror("Error opening file for writing results");
        free(u);
        return;
    }
    char* function_names[] = {"Sequential Full Matrix", "Sequential Half Matrix", "Sequential Swap Vector",
                            "OMP Half Matrix Unroll", "OMP Half Matrix Collapse", "OMP Half Matrix Barrier",
                            "OMP Swap Vector Aligned Tasks", "OMP Half Matrix Aligned", "OMP Swap Vector Aligned",
                            "MPI Half Matrix Broadcast", "MPI Half Matrix Broadcast Matrix", "MPI Half Matrix Scatter Matrix",
                            "MPI Half Matrix Send Recv", "MPI Half Matrix ISend IRecv"};

    fprintf(file, "\n %s \n", function_names[id]);

    for (int i = 0; i < dim; i++) {
        fprintf(file, "      x: %d, u: %.2f\n", i , u[i]);
    }

    fclose(file);
}
