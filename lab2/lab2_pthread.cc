#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

unsigned long long r;
unsigned long long k;
unsigned long long pixels;
pthread_mutex_t lock;

struct data{
	unsigned long long lb;
	unsigned long long rb;
	unsigned long long res;
};


void* my_ceil_sqrl(void* arg){
	struct data* my_data = (struct data*)arg;
	unsigned long long tmp = 0;
	for(unsigned long long x=my_data->lb;x<=my_data->rb;x++){
		tmp += ceil(sqrtl((r+x)*(r-x)));
	}
	pthread_mutex_lock(&lock);
	pixels += tmp;
	if(pixels>=k)
		pixels %= k;
	pthread_mutex_unlock(&lock);
	return NULL;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	pixels = 0;

	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);

	pthread_t my_thread_pool[ncpus];
	struct data my_data[ncpus];
	unsigned long long element_per_thread = std::ceil((r + ncpus - 1) / ncpus);

	pthread_mutex_init(&lock, NULL);

	for(int i=0;i<ncpus;i++){
		my_data[i].res = 0;
		my_data[i].lb = i*element_per_thread;
		if((i+1)*element_per_thread >= r)
			my_data[i].rb = r-1;
		else
			my_data[i].rb = (i+1)*element_per_thread -1;

		if(pthread_create(&my_thread_pool[i], NULL, my_ceil_sqrl, &my_data[i])!=0){
			perror("thread creation failure");
			return 1;
		}
	}

	for(int i=0;i<ncpus;i++){
		pthread_join(my_thread_pool[i], NULL);
	}

	pthread_mutex_destroy(&lock);

	
	printf("%llu\n", (4 * pixels) % k);
}