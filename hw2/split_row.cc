#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

struct data{
    int lb_i;
    int rb_i;
    int j;
    double y0;
};

double left, right, lower, upper;
int width, height, iters, total_cpu, element_per_thread;
int* image;

void* handle_chunk_col(void* arg){
    struct data* my_data = (struct data*)arg;
    #pragma GCC ivdep
    for(int i=my_data->lb_i;i<=my_data->rb_i;i++){
        double x0 = i * ((right - left) / width) + left;
        int repeats = 0;
        double x = 0;
        double y = 0;
        double length_squared = 0;
        while (repeats < iters && length_squared < 4) {
            double temp = x * x - y * y + x0;
            y = 2 * x * y + my_data->y0;
            x = temp;
            length_squared = x * x + y * y;
            ++repeats;
        }
        image[my_data->j * width + i] = repeats;
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

    total_cpu = CPU_COUNT(&cpu_set);
    element_per_thread = std::ceil((width + CPU_COUNT(&cpu_set) - 1) / CPU_COUNT(&cpu_set));
    pthread_t my_thread_pool[total_cpu];
    struct data my_data[total_cpu];

    /* mandelbrot set */
    for (int j = 0; j < height; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for(int i = 0;i<total_cpu; i++){
            my_data[i].lb_i = i*element_per_thread;
            if((i+1)*element_per_thread >= width)
                my_data[i].rb_i = width-1;
            else
                my_data[i].rb_i = (i+1)*element_per_thread -1;
            my_data[i].j = j;
            my_data[i].y0 = y0;
            if(pthread_create(&my_thread_pool[i], NULL, handle_chunk_col, &my_data[i])!=0){
                perror("thread creation failure");
                return 1;
            }
        }


        for(int i=0;i<total_cpu;i++){
            pthread_join(my_thread_pool[i], NULL);
        }
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}