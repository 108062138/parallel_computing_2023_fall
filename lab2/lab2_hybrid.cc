#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);// total number of elements in the series
	unsigned long long k = atoll(argv[2]);// moder
	unsigned long long pixels = 0;

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(size==1){//single thread
		for(int x=0;x<r;x++){
			pixels += ceil(sqrtl(r*r-x*x));
			pixels %= k;
		}
		printf("%llu\n", (pixels*4)%k);
		MPI_Finalize();
		return 0;
	}
	unsigned long long element_per_process = r/(size-1);//one of the thread is used to combine
	unsigned long long start_index = (rank-1) * element_per_process;
	unsigned long long end_index = start_index + element_per_process - 1;
	
	unsigned long long local_pixel = 0;
	unsigned long long x = 0, psum = 0, element_per_thread, lb, rb;
	int omp_threads, omp_thread;

	if(rank==0){
		for(int src=1;src<size;src++){
			unsigned long long rcv_pixel = 0;
			MPI_Recv(&rcv_pixel,1,MPI_UNSIGNED_LONG_LONG,src,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			pixels += rcv_pixel;
			pixels %= k;
		}
		//handle the rest of the execution
		//printf("at last rank %d, range: [ %d, %d] and r: %d, k:%d, elp: %d\n",rank, element_per_process*(size-1), r-1,r,k, r-start_index);
		#pragma parallel shared(local_pixel, element_per_process, start_index, end_index) private(x, psum, lb, rb, element_per_thread)
		{
			omp_threads = omp_get_num_threads();
			omp_thread = omp_get_thread_num();
			element_per_thread = std::ceil((r-start_index) / omp_threads);
			lb = element_per_process*(size-1) + omp_thread*element_per_thread;
			if(lb + (omp_thread+1) * element_per_thread >= r)
				rb = r-1;
			else
				rb = lb + (omp_thread+1) * element_per_thread -1;
			
			//printf(">>>      inside %d, thread: [%d %d] outof thread_id: %d / %d\n", rank, lb, rb, omp_thread, omp_threads);

			#pragma omp parallel for reduction(+: psum)
			for (x = lb; x <= rb; x++) {
				psum += ceil(sqrtl(r*r - x*x));
			}
			if(psum>=k)
				psum %= k;
			#pragma omp critical
			{
				//printf("Thread %d: %d and pixels: %d\n", omp_get_thread_num(), psum, pixels);
				local_pixel += psum;
				if(local_pixel>=k)
					local_pixel %= k;
			}
		}

		pixels += local_pixel;
		if(pixels>=k)
			pixels %= k;
		printf("%llu\n", (pixels*4)%k);
	}else{
		//printf("at rank %d, range: [ %d, %d] and r: %d, k:%d, elp: %d\n",rank, start_index, end_index,r,k, element_per_process);
		
		#pragma parallel shared(local_pixel, element_per_process, start_index, end_index) private(x, psum, lb, rb, element_per_thread)
		{
			omp_threads = omp_get_num_threads();
			omp_thread = omp_get_thread_num();
			element_per_thread = std::ceil((element_per_process + omp_threads - 1) / omp_threads);
			lb = start_index + omp_thread*element_per_thread;
			if(lb + (omp_thread+1) * element_per_thread >= end_index)
				rb = end_index;
			else
				rb = lb + (omp_thread+1) * element_per_thread -1;
			//printf(">>>      inside %d, thread: [%d %d] outof thread_id: %d / %d\n", rank, lb, rb, omp_thread, omp_threads);

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
				local_pixel += psum;
				if(local_pixel>=k)
					local_pixel %= k;
			}
		}

		MPI_Send(&local_pixel,1, MPI_UNSIGNED_LONG_LONG,0,0,MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}