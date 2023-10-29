#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

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
	

	if(rank==0){
		//
		for(int src=1;src<size;src++){
			unsigned long long rcv_pixel = 0;
			MPI_Recv(&rcv_pixel,1,MPI_UNSIGNED_LONG_LONG,src,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			pixels += rcv_pixel;
			pixels %= k;
		}
		//handle the rest of the execution
		//printf("at rank %d, range: [ %d, %d] and r: %d, k:%d, elp: %d\n",rank, element_per_process*(size-1), r-1,r,k, element_per_process);
		for(unsigned long long x=element_per_process*(size-1);x<r;x++){
			pixels += ceil(sqrtl((r+x)*(r-x)));
			
		}
		pixels %= k;
		printf("%llu\n", (pixels*4)%k);
	}else{
		//printf("at rank %d, range: [ %d, %d] and r: %d, k:%d, elp: %d\n",rank, start_index, end_index,r,k, element_per_process);
		unsigned long long local_pixel = 0;
		for(unsigned long long x=start_index;x<=end_index;x++){
			local_pixel += ceil(sqrtl((r+x)*(r-x)));
			
		}
		local_pixel %= k;
		MPI_Send(&local_pixel,1, MPI_UNSIGNED_LONG_LONG,0,0,MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}