#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long psum = 0, element_per_thread = 0, lb, rb, x;
	int omp_threads, omp_thread;

	#pragma omp parallel shared(pixels) private(psum, element_per_thread, lb, rb, x)
	{
		omp_threads = omp_get_num_threads();
		omp_thread = omp_get_thread_num();
		element_per_thread = std::ceil((r + omp_threads - 1) / omp_threads);
		lb = omp_thread*element_per_thread;
		if((omp_thread+1)*element_per_thread >= r)
			rb = r-1;
		else
			rb = (omp_thread+1)*element_per_thread -1;
		psum = 0;

		#pragma omp parallel for reduction(+: psum) 
		for (x = lb; x <= rb; x++) {
			psum += ceil(sqrtl(r*r - x*x));
		}
		if(psum>=k)
			psum %= k;

		#pragma omp critical
		{
			//printf("Thread %d: %d and pixels: %d\n", omp_get_thread_num(), psum, pixels);
			pixels += psum;
		}
	}
	printf("%llu\n", (4 * pixels) % k);
}
