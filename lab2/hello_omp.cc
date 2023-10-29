#include <stdio.h>

#include <omp.h>

int main(int argc, char** argv) {
    int omp_threads, omp_thread;
    int k;

#pragma omp parallel
    {
        omp_threads = omp_get_num_threads();
        omp_thread = omp_get_thread_num();
        k = omp_threads * omp_thread;
        printf("Hello: thread %2d/%2d, k:%d\n", omp_thread, omp_threads, k);
    }
    return 0;
}
