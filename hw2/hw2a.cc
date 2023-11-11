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
#include <immintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <time.h>

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
    //printf("i =  %d,  [%d , %d)\n", my_data->i, my_data->low_j, my_data->high_j);
    // apply vectorization
    __m128d y0, x, y, length_squared, temp, four, zero_vector, repeats;
    __m128d check_iter, check_length, final_check;
    
    int k = my_data->low_j;
    int upper_goto, lower_goto;
    int total_element = my_data->high_j - my_data->low_j, consume_element = 0;
    bool init = true;
    while (total_element > consume_element){
        // initialize
        if(init){
            y0 = _mm_set_pd((k+1) * ((upper - lower) / height) + lower, k * ((upper - lower) / height) + lower);
            upper_goto = k+1;
            lower_goto = k;
            repeats = _mm_set_pd(0, 0);
            x = _mm_set_pd(0, 0);
            y = _mm_set_pd(0, 0);
            length_squared = _mm_set_pd(0, 0);
            four = _mm_set_pd(4, 4);
            zero_vector = _mm_set_pd(0, 0);
            init = false;
            continue;
        }

        // enter while loop, keep consuming elements
        check_iter = _mm_cmplt_pd(repeats, _mm_set_pd(iters, iters));
        check_length = _mm_cmplt_pd(length_squared, four);
        final_check = _mm_and_pd(check_iter, check_length);

        if(_mm_movemask_pd(final_check) == 0x0){
            // fetch next two elements
            double arr[2];
            _mm_store_pd(arr, repeats);
            if(upper_goto < my_data->high_j){
                image[upper_goto * width + my_data->i] = (int)arr[1];
                consume_element++;
                // print out k, i, repeats, length_squared
                //printf("k = %d, at %d, repeats = %d, sqr = %lf\n", upper_goto, upper_goto*width + my_data->i, (int)arr[1], _mm_cvtsd_f64(length_squared));
            }
            if(lower_goto < my_data->high_j){
                image[lower_goto * width + my_data->i] = (int)arr[0];
                consume_element++;
                // print out k, i, repeats, length_squared
                //printf("k = %d, at %d, repeats = %d, sqr = %lf\n", lower_goto, lower_goto*width + my_data->i, (int)arr[0], _mm_cvtsd_f64(length_squared));
            }
            // find new upper and lower goto
            int z = std::max(upper_goto, lower_goto);
            upper_goto = z+2;
            lower_goto = z+1;

            // initialize
            y0 = _mm_set_pd(upper_goto*((upper - lower) / height) + lower, lower_goto*((upper - lower) / height) + lower);
            repeats = _mm_set_pd(0, 0);
            x = _mm_set_pd(0, 0);
            y = _mm_set_pd(0, 0);
            length_squared = _mm_set_pd(0, 0);
            // may conditially break...
        }else if(_mm_movemask_pd(final_check) == 0x1){
            // 01, fetch next one element
            double arr[2];
            _mm_store_pd(arr, repeats);
            // only write the upper one
            if(upper_goto < my_data->high_j){
                image[upper_goto * width + my_data->i] = (int)arr[1];
                consume_element += 1;
                // print out k, i, repeats, length_squared
                //printf("k = %d, at %d, repeats = %d, sqr = %lf\n", upper_goto, upper_goto*width + my_data->i, (int)arr[1], _mm_cvtsd_f64(length_squared));
            }
            // find new upper
            upper_goto = std::max(upper_goto, lower_goto) + 1;
            // partial update e1, for final check == 01
            // consider x, y, y0, repeats.
            // keep all their lower part
            // update upper part of x, y, y0 according to requirement
            // do nothing on upper part of length_squared
            // set upper part of repeats to 0

            y0 = _mm_set_pd(upper_goto*((upper - lower) / height) + lower, lower_goto*((upper - lower) / height) + lower);
            x = _mm_move_sd( zero_vector, x);
            y = _mm_move_sd( zero_vector, y);
            repeats = _mm_move_sd(zero_vector, repeats);

        }else if(_mm_movemask_pd(final_check) == 0x2){
            // 10, fetch next one element
            double arr[2];
            _mm_store_pd(arr, repeats);
            // only write the lower one
            if(lower_goto < my_data->high_j){
                image[lower_goto * width + my_data->i] = (int)arr[0];
                consume_element += 1;
                // print out k, i, repeats, length_squared
                //printf("k = %d, at %d, repeats = %d, sqr = %lf\n", lower_goto, lower_goto*width + my_data->i, (int)arr[0], _mm_cvtsd_f64(length_squared));
            }
            // find new lower
            lower_goto = std::max(upper_goto, lower_goto) + 1;
            // partial update e0, for final check == 10
            y0 = _mm_set_pd(upper_goto*((upper - lower) / height) + lower, lower_goto*((upper - lower) / height) + lower);
            x = _mm_move_sd(x, zero_vector);
            y = _mm_move_sd(y, zero_vector);
            repeats = _mm_move_sd(repeats, zero_vector);
        }

        // update
        temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), _mm_set_pd(my_data->x0, my_data->x0));
        y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(_mm_set_pd(2, 2), x), y), y0);
        x = temp;
        length_squared = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
        repeats = _mm_add_pd(repeats, _mm_set_pd(1, 1));
    }
    

    return NULL;
    
    //for(int k=my_data->low_j;k<my_data->high_j;k++){
    //    double y0 = k * ((upper - lower) / height) + lower;
    //    int repeats = 0;
    //    double x = 0;
    //    double y = 0;
    //    double length_squared = 0;
    //    while (repeats < iters && length_squared < 4) {
    //        double temp = x * x - y * y + my_data->x0;
    //        y = 2 * x * y + y0;
    //        x = temp;
    //        length_squared = x * x + y * y;
    //        ++repeats;
    //    }
    //    image[k * width + my_data->i] = repeats;
    //    printf("k = %d, at %d, repeats = %d, sqr = %lf\n", k, k*width + my_data->i, repeats, length_squared);
    //}
    //return NULL;
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
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));

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
    //printf("total length: %d\n", width*height);
    total_cpu = CPU_COUNT(&cpu_set);
    pthread_t threads[total_cpu];
    struct data my_data[total_cpu];
    
    float* each_thread_time = (float*)calloc(total_cpu, sizeof(float));
    struct timespec start_each[total_cpu], end_each[total_cpu];

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
            my_data[j].low_j = partition_left[j];
            my_data[j].high_j = partition_right[j];
            my_data[j].i = i;
            my_data[j].x0 = x0;
            clock_gettime(CLOCK_MONOTONIC_RAW, &start_each[j]);
            pthread_create(&threads[j], NULL, handle_chunk_row, (void*)&my_data[j]);
        }

        for(int j=0;j<total_cpu;j++){
            pthread_join(threads[j], NULL);
            clock_gettime(CLOCK_MONOTONIC_RAW, &end_each[j]);
            each_thread_time[j] += end_each[j].tv_sec - start_each[j].tv_sec + (end_each[j].tv_nsec - start_each[j].tv_nsec)/1000000000.0;
        }
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    free(prefix_sum);
    free(partition_left);
    free(partition_right);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    // printf("time: %lf sec\n", end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec)/1000000000.0);
    // for(int i=0;i<total_cpu;i++){
    //     printf("thread %d time: %lf sec\n", i, each_thread_time[i]);
    // }
}