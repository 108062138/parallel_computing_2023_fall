#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

typedef struct data {
    int low_j;
    int high_j;
    int i;
    double x0;
} data;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Hello from rank %d of %d\n", rank, size);

    // Define the MPI data type for the 'data' struct
    int block_lengths[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};
    MPI_Aint displacements[4];
    data sampleData;

    MPI_Get_address(&sampleData.low_j, &displacements[0]);
    MPI_Get_address(&sampleData.high_j, &displacements[1]);
    MPI_Get_address(&sampleData.i, &displacements[2]);
    MPI_Get_address(&sampleData.x0, &displacements[3]);

    // Calculate displacements
    MPI_Aint base = displacements[0];
    for (int i = 0; i < 4; i++) {
        displacements[i] -= base;
    }

    MPI_Datatype myDataType;
    MPI_Type_create_struct(4, block_lengths, displacements, types, &myDataType);
    MPI_Type_commit(&myDataType);

    // Sending and receiving data
    if (size > 1) {
        if (rank == 0) {
            data data_to_send;
            for(int i=1;i<size;i++){
                data_to_send.low_j = 0;
                data_to_send.high_j = 10;
                data_to_send.i = i;
                data_to_send.x0 = 1.2-i;
                MPI_Send(&data_to_send, 1, myDataType, i, 0, MPI_COMM_WORLD);
            }
        } else {
            data received_data;
            MPI_Recv(&received_data, 1, myDataType, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Received: low_j=%d, high_j=%d, i=%d, x0=%lf\n",
                received_data.low_j, received_data.high_j, received_data.i, received_data.x0);
        }
    }

    MPI_Type_free(&myDataType);
    MPI_Finalize();

    return 0;
}