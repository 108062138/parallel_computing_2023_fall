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

struct data{
    int low_j;
    int high_j;
    int i;
    double x0;
};

int iters, width, height, element_per_thread, total_cpu;
double left, right, lower, upper;
int* image;

void* handle_chunk_row(void* arg){
    struct data* my_data = (struct data*)arg;
    for(int k=my_data->low_j;k<my_data->high_j;k++){
        double y0 = k * ((upper - lower) / height) + lower;
        int repeats = 0;
        double x = 0;
        double y = 0;
        double length_squared = 0;
        while (repeats < iters && length_squared < 4) {
            double temp = x * x - y * y + my_data->x0;
            y = 2 * x * y + y0;
            x = temp;
            length_squared = x * x + y * y;
            ++repeats;
        }
        image[k * width + my_data->i] = repeats;
    }
    return NULL;
}

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
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));

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

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    element_per_thread = std::ceil((height + CPU_COUNT(&cpu_set) - 1) / CPU_COUNT(&cpu_set));
    printf("total length: %d\n", width*height);
    total_cpu = CPU_COUNT(&cpu_set);
    pthread_t threads[total_cpu];
    struct data my_data[total_cpu];
    int* prefix_sum = (int*)malloc(height * sizeof(int));
    int* partition_left = (int*)malloc(total_cpu * sizeof(int));
    int* partition_right = (int*)malloc(total_cpu * sizeof(int));
    /* mandelbrot set */
    for (int i = 0; i < width; ++i) {
        double x0 = i * ((right - left) / width) + left;
        if(i!=0){
            // calculate the previous column's prefix sum
            prefix_sum[0] = image[0*width+i-1];
            for(int j=1;j<height;j++){
                prefix_sum[j] = prefix_sum[j-1] + image[j*width+i-1];
            }
            int current_index = 0, next_index = 0, current_acc = 0, next_acc = 0;
            int average = prefix_sum[height-1]/total_cpu;
            for(int j=0;j<total_cpu;j++){
                if(j==total_cpu-1){
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
            //for(int j=0;j<total_cpu;j++){
            //    printf("partition %d: %d - %d\n", j, partition_left[j], partition_right[j]);
            //}
        }else{
            // handle 0th column
            for(int j=0;j<total_cpu;j++){
                partition_left[j] = j * element_per_thread;
                if((j+1)*element_per_thread >= height)
                    partition_right[j] = height;
                else
                    partition_right[j] = (j+1)*element_per_thread;
            }
        }
        for(int j=0;j<total_cpu;j++){
            //int low_j = j * element_per_thread, high_j;
            //if ((j+1)*element_per_thread >= height)
            //    high_j = height;
            //else
            //    high_j = (j+1)*element_per_thread;
            //for(int k=low_j;k<=high_j;k++){
            //    double y0 = k * ((upper - lower) / height) + lower;
            //    int repeats = 0;
            //    double x = 0;
            //    double y = 0;
            //    double length_squared = 0;
            //    while (repeats < iters && length_squared < 4) {
            //        double temp = x * x - y * y + x0;
            //        y = 2 * x * y + y0;
            //        x = temp;
            //        length_squared = x * x + y * y;
            //        ++repeats;
            //    }
            //    image[k * width + i] = repeats;
            //}

            my_data[j].low_j = partition_left[j];
            my_data[j].high_j = partition_right[j];
            my_data[j].i = i;
            my_data[j].x0 = x0;
            pthread_create(&threads[j], NULL, handle_chunk_row, (void*)&my_data[j]);
        }

        for(int j=0;j<total_cpu;j++)
            pthread_join(threads[j], NULL);

        //for (int j = 0; j < height; ++j) {
        //    double y0 = j * ((upper - lower) / height) + lower;
        //    int repeats = 0;
        //    double x = 0;
        //    double y = 0;
        //    double length_squared = 0;
        //    while (repeats < iters && length_squared < 4) {
        //        double temp = x * x - y * y + x0;
        //        y = 2 * x * y + y0;
        //        x = temp;
        //        length_squared = x * x + y * y;
        //        ++repeats;
        //    }
        //    image[j * width + i] = repeats;
        //}
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
