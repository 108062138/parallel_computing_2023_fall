#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <algorithm>
#include <mpi.h>
#include <omp.h>

#define TRANSFER_DATA_TAG 123
#define TRANSFER_PIXEL_TAG 456


typedef struct data{
    int low_j;
    int high_j;
    int i;
    double x0;
} data;

int iters, width, height, element_per_node, total_cpu;
double left, right, lower, upper;
int* image;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many nodes are available */
    MPI_Init(&argc, &argv);
    int rank, size;
    double start_time, end_time, cpu_time=0, io_time=0, comm_time=0;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //printf("Hello from rank %d of %d\n", rank, size);

    // set up data type in MPI
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

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    if(rank==0 && size==1){
        image = (int*)malloc(width * height * sizeof(int));
        start_time = MPI_Wtime();
        /* mandelbrot set */
        for (int j = 0; j < height; ++j) {
            double y0 = j * ((upper - lower) / height) + lower;
            for (int i = 0; i < width; ++i) {
                double x0 = i * ((right - left) / width) + left;
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                image[j * width + i] = repeats;
            }
        }

        /* draw and cleanup */
        write_png(filename, iters, width, height, image);
        free(image);
        MPI_Finalize();
        end_time = MPI_Wtime();
        printf("single node: cpu time: %lf, io time: %lf, comm time: %lf\n", end_time-start_time, 0.0, 0.0);
        return 0;
    }

    /* allocate memory for image */
    if(rank==0){
        image = (int*)malloc(width * height * sizeof(int));
        assert(image);
    }
    element_per_node = std::ceil((height + size - 1 - 1) / (size - 1));
    //printf("total length: %d\n", width*height);
    data my_data[size];
    int* prefix_sum = (int*)malloc(height * sizeof(int));
    int* partition_left = (int*)malloc(size * sizeof(int));
    int* partition_right = (int*)malloc(size * sizeof(int));
    int* single_column = (int*)malloc(height * sizeof(int));

    /* mandelbrot set */
    for (int i = 0; i < width; ++i) {
        double x0 = i * ((right - left) / width) + left;
        // early stop for single node
        if(rank==0 && size==1){
            printf("single node\n");
            start_time = MPI_Wtime();
            #pragma parallel shared(x0, i, image)
            {
                int omp_threads = omp_get_num_threads();
                int omp_thread = omp_get_thread_num();
                int element_per_thread = std::ceil((height + omp_threads - 1) / omp_threads);
                int lb = omp_thread*element_per_thread;
                int rb = std::min(height, lb + element_per_thread);
                #pragma omp parallel for schedule(dynamic)
                for(int k=lb;k<rb;k++){
                    double y0 = k * ((upper - lower) / height) + lower;
                    int repeats = 0;
                    double x = 0;
                    double y = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    image[k * width + i] = repeats;
                }
            }
            end_time = MPI_Wtime();
            cpu_time += end_time - start_time;
            continue;
        }
        // naive split the work
        if(rank==0){
            // first calculate partition left and right
            start_time = MPI_Wtime();
            if(i==0){//naive partition
                for(int j=1;j<size;j++){
                    partition_left[j] = (j-1) * element_per_node;
                    partition_right[j] = std::min(j * element_per_node, height);
                }
            }else{//start load balance
                prefix_sum[0] = image[0*width+i-1];
                for(int j=1;j<height;j++){
                    prefix_sum[j] = prefix_sum[j-1] + image[j*width+i-1];
                }
                int current_index = 0, next_index = 0, current_acc = 0, next_acc = 0;
                int average = prefix_sum[height-1]/(size-1);
                for(int j=1;j<size;j++){
                    if(j==size-1){
                        partition_left[j] = current_index;
                        partition_right[j] = height;
                    }else{
                        next_acc = current_acc + average;
                        int* tmp = std::lower_bound(prefix_sum+current_index, prefix_sum+height, next_acc);
                        next_index = tmp - prefix_sum;
                        partition_left[j] = current_index;
                        partition_right[j] = next_index;

                        current_index = next_index;
                        current_acc = next_acc;
                    }
                }
            }
            end_time = MPI_Wtime();
            cpu_time += end_time - start_time;
            
            // then send data to each node
            start_time = MPI_Wtime();
            for(int j=1;j<size;j++){
                my_data[j].low_j = partition_left[j];
                my_data[j].high_j = partition_right[j];
                my_data[j].i = i;
                my_data[j].x0 = x0;
                MPI_Send(   &my_data[j], 1, myDataType, j, TRANSFER_DATA_TAG,
                            MPI_COMM_WORLD);
            }
            // collect pixel data from each node
            for(int j=1;j<size;j++){
                MPI_Recv(   single_column, height, MPI_INT, j, TRANSFER_PIXEL_TAG,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                for(int k=my_data[j].low_j;k<my_data[j].high_j;k++){
                    image[k * width + my_data[j].i] = single_column[k];
                }
            }
            end_time = MPI_Wtime();
            comm_time += end_time - start_time;
        }else{
            data received_data;
            // receive data from master
            start_time = MPI_Wtime();
            MPI_Recv(   &received_data, 1, myDataType, 0, TRANSFER_DATA_TAG,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            end_time = MPI_Wtime();
            comm_time += end_time - start_time;
            //printf("at iter %d, under rank %d, received: low_j=%d, high_j=%d, i=%d, x0=%lf\n",
            //    i, rank, received_data.low_j, received_data.high_j, received_data.i, received_data.x0);
            // reset single column as -1
            for(int j=0;j<height;j++){
                single_column[j] = -1;
            }
            // perform calculation
            int omp_threads, omp_thread;
            start_time = MPI_Wtime();
            #pragma parallel shared(received_data, single_column) private(omp_threads, omp_thread)
            {
                omp_threads = omp_get_num_threads();
                omp_thread = omp_get_thread_num();
                int element_per_thread = std::ceil((received_data.high_j - received_data.low_j + omp_threads - 1) / omp_threads);
                int lb = received_data.low_j + omp_thread*element_per_thread;
                int rb = std::min(received_data.high_j, lb + element_per_thread);
                
                #pragma omp parallel for schedule(dynamic)
                for(int k=lb;k<rb;k++){
                    double y0 = k * ((upper - lower) / height) + lower;
                    int repeats = 0;
                    double x = 0;
                    double y = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4) {
                        double temp = x * x - y * y + received_data.x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    single_column[k] = repeats;
                }
            }
            end_time = MPI_Wtime();
            cpu_time += end_time - start_time;

            start_time = MPI_Wtime();
            MPI_Send(   single_column, height, MPI_INT, 0, TRANSFER_PIXEL_TAG,
                        MPI_COMM_WORLD);
            end_time = MPI_Wtime();
            comm_time += end_time - start_time;
        }
    }

    /* draw and cleanup */
    if (rank == 0){
        start_time = MPI_Wtime();
        write_png(filename, iters, width, height, image);
        end_time = MPI_Wtime();
        io_time += end_time - start_time;
    }
    // printf("rank %d: cpu time: %lf, io time: %lf, comm time: %lf\n", rank, cpu_time, io_time, comm_time);
        
    // if (rank == 0)
    //     printf("rank 0: cpu time: %lf, io time: %lf, comm time: %lf\n", cpu_time, io_time, comm_time);
    printf("rank %d: cpu time: %lf, io time: %lf, comm time: %lf\n", rank, cpu_time, io_time, comm_time);
    free(image);
    free(prefix_sum);
    free(partition_left);
    free(partition_right);
    MPI_Finalize();
}