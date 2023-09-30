#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_elements = 1000;  // Total number of elements in the series
    int elements_per_process = num_elements / size;  // Elements per process

    double local_value, local_result;

    // Compute the start and end indices for this process
    int start_index = rank * elements_per_process + 1;
    int end_index = start_index + elements_per_process - 1;

    // Compute the square root locally
    for (int i = start_index; i <= end_index; i++) {
        local_value = i;
        local_result = sqrt(local_value);
        printf("Process %d: sqrt(%d) = %f\n", rank, i, local_result);
    }

    // Collect results at the root process (rank 0)
    if (rank != 0) {
        MPI_Send(&local_result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        double gathered_result;
        for (int source = 1; source < size; source++) {
            MPI_Recv(&gathered_result, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d received result %f from process %d\n", rank, gathered_result, source);
        }
    }

    MPI_Finalize();

    return 0;
}
