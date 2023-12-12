#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <math.h>
#include <iomanip>  // for std::setw
#include <cstdio>   // for fread
#include <omp.h>


#define NOT_REACHABLE         (1073741823)
#define BASIC_WARP            (32)
#define COALESCED_FACTOR      (BASIC_WARP*4)
#define BLOCKING_FACTOR   (BASIC_WARP*2)
// change this to change thread setting
#define THREAD_XY_INTERACTION (2)
// THREAD_X_DIM * THREAD_Y_DIM = 1024
#define THREAD_X_DIM          (BASIC_WARP/THREAD_XY_INTERACTION)
#define THREAD_Y_DIM          (BASIC_WARP*THREAD_XY_INTERACTION)
#define THREAD_LOAD_SAVE_NUM  (BLOCKING_FACTOR/THREAD_X_DIM)

using namespace std;

int maxWidth;
int num_vertex, num_edge, num_blocked_square_row, num_blocked_square_col;
int padded_num_vertex; // extend to fit COALESCED_FACTOR
int *dist; // one dimension array, lenght = padded_num_vertex * padded_num_vertex
int *rem_orig_dist;

void input(char* input_file) {
    FILE* file = fopen(input_file, "rb");
    fread(&num_vertex, sizeof(int), 1, file);
    fread(&num_edge, sizeof(int), 1, file);
    // pad the matrix to fit COALESCED_FACTOR
    int remainder = num_vertex % COALESCED_FACTOR;
    padded_num_vertex = (remainder == 0) ? num_vertex : num_vertex + (COALESCED_FACTOR - remainder);
    dist = (int*)malloc(padded_num_vertex * padded_num_vertex * sizeof(int));
    //rem_orig_dist = (int*)malloc(padded_num_vertex * padded_num_vertex * sizeof(int));
    // initialize dist
    for (int i = 0; i < padded_num_vertex; ++i){ 
        for (int j = 0; j < padded_num_vertex; ++j){
            dist[i * padded_num_vertex + j] = (i == j) ? 0 : NOT_REACHABLE;
            //rem_orig_dist[i * padded_num_vertex + j] = (i == j) ? 0 : NOT_REACHABLE;
        }
    }
    // read file
    int pair[3];
    for (int i = 0; i < num_edge; ++i) {
        fread(pair, sizeof(int), 3, file);
        int from = pair[0], to = pair[1], weight = pair[2];
        dist[from * padded_num_vertex + to] = weight;
        //rem_orig_dist[from * padded_num_vertex + to] = weight;
    }
    fclose(file);
}
void output(char* output_file) {
    FILE* outfile = fopen(output_file, "w");
    // unpad the matrix
    for (int i = 0; i < num_vertex; ++i) 
        for (int j = 0; j < num_vertex; ++j)
            // since we padded the matrix, we have to move the result back
            // since we travel the matrix in row-major order, we will not mess up the result by using the stride of padded_num_vertex
            dist[i * num_vertex + j] = (dist[i * padded_num_vertex + j] >= NOT_REACHABLE) ? NOT_REACHABLE : dist[i * padded_num_vertex + j];
    // use fwrite to speed up output
    fwrite(dist, sizeof(int), num_vertex * num_vertex, outfile);
    fclose(outfile);
}

__global__ void phase_1(int* d_dist, int padded_num_vertex, int round){
    int glb_block_i = round;
    int glb_block_j = round;
    int in_block_i = threadIdx.y;
    int in_block_j = threadIdx.x * THREAD_LOAD_SAVE_NUM;
    // load gloval data into my basic block's share memory
    extern __shared__ int share_target[BLOCKING_FACTOR][BLOCKING_FACTOR];
    #pragma unroll 4
    for(int id = 0; id < THREAD_LOAD_SAVE_NUM; id++){
        share_target[in_block_i][in_block_j + id] =  d_dist[(glb_block_i*BLOCKING_FACTOR+in_block_i) * padded_num_vertex + (glb_block_j * BLOCKING_FACTOR + in_block_j) + id];
    }
    __syncthreads();
    // start computing and enumerate k
    for(int k=0;k<BLOCKING_FACTOR;k++){
        #pragma unroll 4
        for(int id = 0; id<THREAD_LOAD_SAVE_NUM; id++){
            //new from in_block_i to in_block_j+id          old from in_block_i to in_block_j+id                    from  in_block_i to k, from k to in_block_j+id
            share_target[in_block_i][in_block_j + id] = min(
                share_target[in_block_i][in_block_j + id],
                share_target[in_block_i][k] + share_target[k][in_block_j + id]);
        }
        // there are dependence inbetween k at phase 1
        __syncthreads();
    }
    // write back
    #pragma unroll 4
    for(int id = 0; id < THREAD_LOAD_SAVE_NUM; id++){
        d_dist[(glb_block_i*BLOCKING_FACTOR+in_block_i) * padded_num_vertex + (glb_block_j * BLOCKING_FACTOR + in_block_j) + id] = share_target[in_block_i][in_block_j + id];
    }
}

__global__ void phase_2_fuse(int* d_dist, int padded_num_vertex, int round){
    if(round == blockIdx.x) return;
    int glb_b_i_same_row = round;
    int glb_b_j_same_row = blockIdx.x;
    int glb_b_i_same_col = blockIdx.x;
    int glb_b_j_same_col = round;
    int in_block_i = threadIdx.y;
    int in_block_j = threadIdx.x * THREAD_LOAD_SAVE_NUM;

    // load data
    extern __shared__ int from_same_row[BLOCKING_FACTOR][BLOCKING_FACTOR];
    extern __shared__ int from_same_col[BLOCKING_FACTOR][BLOCKING_FACTOR];
    extern __shared__ int to[BLOCKING_FACTOR][BLOCKING_FACTOR];

    for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
        from_same_row[in_block_i][in_block_j+id] = d_dist[(glb_b_i_same_row*BLOCKING_FACTOR+in_block_i)*padded_num_vertex + (glb_b_j_same_row*BLOCKING_FACTOR + in_block_j) + id];
        from_same_col[in_block_i][in_block_j+id] = d_dist[(glb_b_i_same_col*BLOCKING_FACTOR+in_block_i)*padded_num_vertex+(glb_b_j_same_col*BLOCKING_FACTOR+in_block_j) + id];
        to[in_block_i][in_block_j+id]   = d_dist[(round*BLOCKING_FACTOR+in_block_i)* padded_num_vertex + (round*BLOCKING_FACTOR+in_block_j) + id];
    }
    __syncthreads();
    for(int k=0;k<BLOCKING_FACTOR;k++){
        #pragma unroll 4
        for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
            from_same_row[in_block_i][in_block_j+id] = min(
                from_same_row[in_block_i][in_block_j+id],
                to[in_block_i][k] + from_same_row[k][in_block_j+id]
            );
            from_same_col[in_block_i][in_block_j+id] = min(
                from_same_col[in_block_i][in_block_j+id],
                from_same_col[in_block_i][k] + to[k][in_block_j+id]
            );
        }
        __syncthreads();
    }
    for(int id=0;id < THREAD_LOAD_SAVE_NUM;id++){
        d_dist[(glb_b_i_same_row*BLOCKING_FACTOR + in_block_i) * padded_num_vertex + (glb_b_j_same_row*BLOCKING_FACTOR+in_block_j) + id] = from_same_row[in_block_i][in_block_j+id];
        d_dist[(glb_b_i_same_col*BLOCKING_FACTOR + in_block_i) * padded_num_vertex + (glb_b_j_same_col*BLOCKING_FACTOR+in_block_j) + id] = from_same_col[in_block_i][in_block_j+id];
    }
}

__global__ void phase_2_same_row(int* d_dist, int padded_num_vertex, int round){
    // for now, only ha
    if(round == blockIdx.x) return;
    int glb_block_i = round;
    int glb_block_j = blockIdx.x;
    int in_block_i = threadIdx.y;
    int in_block_j = threadIdx.x * THREAD_LOAD_SAVE_NUM;

    // load data
    extern __shared__ int from[BLOCKING_FACTOR][BLOCKING_FACTOR];
    extern __shared__ int to[BLOCKING_FACTOR][BLOCKING_FACTOR];
    #pragma unroll 4
    for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
        from[in_block_i][in_block_j+id] = d_dist[(glb_block_i*BLOCKING_FACTOR+in_block_i)*padded_num_vertex + (glb_block_j*BLOCKING_FACTOR + in_block_j) + id];
        to[in_block_i][in_block_j+id]   = d_dist[(round*BLOCKING_FACTOR+in_block_i)* padded_num_vertex + (round*BLOCKING_FACTOR+in_block_j) + id];
    }
    __syncthreads();

    // cal
    for(int k=0;k<BLOCKING_FACTOR;k++){
        #pragma unroll 4
        for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
            from[in_block_i][in_block_j+id] = min(
                from[in_block_i][in_block_j+id],
                to[in_block_i][k] + from[k][in_block_j+id]
            );
        }
        __syncthreads();
    }
    #pragma unroll 4
    for(int id=0;id < THREAD_LOAD_SAVE_NUM;id++){
        d_dist[(glb_block_i*BLOCKING_FACTOR + in_block_i) * padded_num_vertex + (glb_block_j*BLOCKING_FACTOR+in_block_j) + id] = from[in_block_i][in_block_j+id];
    }
}
__global__ void phase_2_same_col(int* d_dist, int padded_num_vertex, int round){
    if(round == blockIdx.x) return;
    int glb_block_i = blockIdx.x;
    int glb_block_j = round;
    int in_block_i = threadIdx.y;
    int in_block_j = threadIdx.x * THREAD_LOAD_SAVE_NUM;
    // load data
    extern __shared__ int from[BLOCKING_FACTOR][BLOCKING_FACTOR];
    extern __shared__ int to[BLOCKING_FACTOR][BLOCKING_FACTOR];
    #pragma unroll 4
    for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
        from[in_block_i][in_block_j+id]   = d_dist[(glb_block_i*BLOCKING_FACTOR+in_block_i)*padded_num_vertex+(glb_block_j*BLOCKING_FACTOR+in_block_j) + id];
        to[in_block_i][in_block_j+id] = d_dist[(round*BLOCKING_FACTOR+in_block_i)*padded_num_vertex + (round*BLOCKING_FACTOR+in_block_j)+ id];
    }
    __syncthreads();
    
    for(int k=0;k<BLOCKING_FACTOR;k++){
        #pragma unroll 4
        for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
            from[in_block_i][in_block_j+id] = min(
                from[in_block_i][in_block_j+id],
                from[in_block_i][k] + to[k][in_block_j+id]
            );
        }
        __syncthreads();
    }
    #pragma unroll 4
    for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
        d_dist[(glb_block_i*BLOCKING_FACTOR + in_block_i) * padded_num_vertex + (glb_block_j*BLOCKING_FACTOR+in_block_j) + id] = from[in_block_i][in_block_j+id];
    }
}

__global__ void phase_3(int* d_dist, int padded_num_vertex, int round, int offset){
    if(round == blockIdx.x || round == (blockIdx.y+offset/BLOCKING_FACTOR)) return;
    int glb_block_i = blockIdx.y;
    int glb_block_j = blockIdx.x;
    int in_block_i = threadIdx.y;
    int in_block_j = threadIdx.x * THREAD_LOAD_SAVE_NUM;
    int arr[THREAD_LOAD_SAVE_NUM];
    extern __shared__ int from[BLOCKING_FACTOR][BLOCKING_FACTOR];
    extern __shared__ int to[BLOCKING_FACTOR][BLOCKING_FACTOR];
    // precalculate address
    int src_addr = (glb_block_i*BLOCKING_FACTOR+in_block_i+offset)*padded_num_vertex + (glb_block_j*BLOCKING_FACTOR+in_block_j);//
    int from_addr = (round*BLOCKING_FACTOR+in_block_i)*padded_num_vertex+(glb_block_j*BLOCKING_FACTOR + in_block_j);
    int to_addr = (glb_block_i*BLOCKING_FACTOR+in_block_i+offset)*padded_num_vertex+(round*BLOCKING_FACTOR + in_block_j);//

    // load value
    #pragma unroll
    for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
        // init arr
        arr[id] = d_dist[src_addr + id];
        // load from and to
        from[in_block_i][in_block_j+id] = d_dist[from_addr + id];
        to[in_block_i][in_block_j+id] = d_dist[to_addr + id];
    }
    __syncthreads();
    for(int k=0;k<BLOCKING_FACTOR;k++){
        #pragma unroll
        for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
            arr[id] = min(
                arr[id],
                to[in_block_i][k]+from[k][in_block_j+id]);
        }
    }
    #pragma unroll
    for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
        d_dist[src_addr + id] = arr[id];
    }
}

__global__ void phase_3_a(int *d_dist, int padded_num_vertex, int round, int block_start_row){
    if(round == blockIdx.x || round == blockIdx.y + block_start_row) return;
    int glb_block_i = blockIdx.y + block_start_row;
    int glb_block_j = blockIdx.x;
    int in_block_i = threadIdx.y;
    int in_block_j = threadIdx.x * THREAD_LOAD_SAVE_NUM;
    int arr[THREAD_LOAD_SAVE_NUM];
    extern __shared__ int from[BLOCKING_FACTOR][BLOCKING_FACTOR];
    extern __shared__ int to[BLOCKING_FACTOR][BLOCKING_FACTOR];
    // precalculate address
    int src_addr = (glb_block_i*BLOCKING_FACTOR+in_block_i)*padded_num_vertex + (glb_block_j*BLOCKING_FACTOR+in_block_j);
    int from_addr = (round*BLOCKING_FACTOR+in_block_i)*padded_num_vertex+(glb_block_j*BLOCKING_FACTOR + in_block_j);
    int to_addr = (glb_block_i*BLOCKING_FACTOR+in_block_i)*padded_num_vertex+(round*BLOCKING_FACTOR + in_block_j);

    // load value
    #pragma unroll
    for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
        // init arr
        arr[id] = d_dist[src_addr + id];
        // load from and to
        from[in_block_i][in_block_j+id] = d_dist[from_addr + id];
        to[in_block_i][in_block_j+id] = d_dist[to_addr + id];
    }
    __syncthreads();
    for(int k=0;k<BLOCKING_FACTOR;k++){
        #pragma unroll
        for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
            arr[id] = min(
                arr[id],
                to[in_block_i][k]+from[k][in_block_j+id]);
        }
    }
    #pragma unroll
    for(int id=0;id<THREAD_LOAD_SAVE_NUM;id++){
        d_dist[src_addr + id] = arr[id];
    }
}

// [remote] pp23s80	—	41	249.29		0.32	0.17	0.17	0.17	0.17	0.17	0.17	0.17	0.22	0.22	0.22	0.17	0.17	0.17	0.22	0.32	0.32	0.47	0.47	0.97	1.17	1.77	2.32	2.92	3.62	4.53	5.33	6.18	7.18	7.98	9.18	10.54	11.94	13.62	16.35	17.50	18.96	22.61	23.87	27.43	28.92
void blocked_floyd_warshell(){
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    omp_set_num_threads(num_gpus);
    cudaHostRegister(dist, padded_num_vertex * padded_num_vertex * sizeof(int), cudaHostRegisterDefault);
    // allocate memory for dist in device
    int* d_dist[2];
    int y_offset = ((padded_num_vertex/BLOCKING_FACTOR) % 2 == 0)? padded_num_vertex/2: (padded_num_vertex/BLOCKING_FACTOR/2+1)*BLOCKING_FACTOR;

    #pragma omp parallel
    {
        num_blocked_square_row = padded_num_vertex / BLOCKING_FACTOR;
        num_blocked_square_col = padded_num_vertex / BLOCKING_FACTOR;
        dim3 basic_block(THREAD_X_DIM, THREAD_Y_DIM);
        dim3 phase_1_grid(1);
        dim3 phase_2_grid(num_blocked_square_row);
        dim3 phase_3_grid(num_blocked_square_row, y_offset/BLOCKING_FACTOR);
        
        unsigned int tid = omp_get_thread_num();
        cudaSetDevice(tid);
        cudaMalloc(&d_dist[tid], padded_num_vertex * padded_num_vertex * sizeof(int));
        cudaMemcpy(d_dist[tid], dist, padded_num_vertex * padded_num_vertex * sizeof(int), cudaMemcpyHostToDevice);

        for(int round=0;round<num_blocked_square_row;round++){
            phase_1      <<< phase_1_grid, basic_block>>> (d_dist[tid], padded_num_vertex, round);
            phase_2_fuse <<< phase_2_grid, basic_block>>> (d_dist[tid], padded_num_vertex, round);
            phase_3      <<< phase_3_grid, basic_block>>> (d_dist[tid], padded_num_vertex, round, y_offset*tid);
            cudaDeviceSynchronize();
            #pragma omp barrier
            if(tid==1 && (round+1) < y_offset/BLOCKING_FACTOR){
                cudaMemcpy(d_dist[1] + (round+1)*BLOCKING_FACTOR*padded_num_vertex, d_dist[0] + (round+1)*BLOCKING_FACTOR*padded_num_vertex, BLOCKING_FACTOR*padded_num_vertex*sizeof(int), cudaMemcpyDeviceToDevice);
            }else if(tid==0 && (round+1)>=y_offset/BLOCKING_FACTOR){
                cudaMemcpy(d_dist[0] + (round+1)*BLOCKING_FACTOR*padded_num_vertex, d_dist[1] + (round+1)*BLOCKING_FACTOR*padded_num_vertex, BLOCKING_FACTOR*padded_num_vertex*sizeof(int), cudaMemcpyDeviceToDevice);
            }
        }
        if(tid==0)
            cudaMemcpy(dist, d_dist[tid], y_offset*padded_num_vertex*sizeof(int), cudaMemcpyDeviceToHost);
        else
            cudaMemcpy(dist + y_offset*padded_num_vertex, d_dist[tid] + y_offset*padded_num_vertex, (padded_num_vertex-y_offset)*padded_num_vertex*sizeof(int), cudaMemcpyDeviceToHost);
    }
}

int main(int argc, char* argv[]){
    assert(argc == 3);
    maxWidth = 5;
    char* input_file = argv[1];
    char* output_file = argv[2];
    // parse input file
    input(input_file);
    // blocked floyd-warshall
    blocked_floyd_warshell();
    // output file
    output(output_file);
    return 0;
}