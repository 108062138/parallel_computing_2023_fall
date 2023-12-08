#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <math.h>
#include <iomanip>  // for std::setw
#include <cstdio>   // for fread


#define NOT_REACHABLE         (1073741823)
#define BASIC_WARP            (4) // 32
#define COALESCED_FACTOR      (BASIC_WARP*4)
#define BLOCKED_SQUARE_SIZE   (BASIC_WARP*2)
// change this to change thread setting
#define THREAD_XY_INTERACTION (2)
// THREAD_X_DIM * THREAD_Y_DIM = 1024
#define THREAD_X_DIM          (BASIC_WARP/THREAD_XY_INTERACTION)
#define THREAD_Y_DIM          (BASIC_WARP*THREAD_XY_INTERACTION)
#define THREAD_LOAD_SAVE_NUM  (BLOCKED_SQUARE_SIZE/THREAD_X_DIM)

using namespace std;

int maxWidth;
int num_vertex, num_edge, num_blocked_square_row, num_blocked_square_col;
int padded_num_vertex; // extend to fit COALESCED_FACTOR
int *dist; // one dimension array, lenght = padded_num_vertex * padded_num_vertex
int *rem_orig_dist;

void input(char* input_file) {
    FILE* file = fopen(input_file, "rb");
    // check open file
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", input_file);
        exit(1);
    }
    fread(&num_vertex, sizeof(int), 1, file);
    fread(&num_edge, sizeof(int), 1, file);
    // pad the matrix to fit COALESCED_FACTOR
    int remainder = num_vertex % COALESCED_FACTOR;
    padded_num_vertex = (remainder == 0) ? num_vertex : num_vertex + (COALESCED_FACTOR - remainder);
    dist = (int*)malloc(padded_num_vertex * padded_num_vertex * sizeof(int));
    rem_orig_dist = (int*)malloc(padded_num_vertex * padded_num_vertex * sizeof(int));
    // initialize dist
    for (int i = 0; i < padded_num_vertex; ++i){ 
        for (int j = 0; j < padded_num_vertex; ++j){
            dist[i * padded_num_vertex + j] = (i == j) ? 0 : NOT_REACHABLE;
            rem_orig_dist[i * padded_num_vertex + j] = (i == j) ? 0 : NOT_REACHABLE;
        }
    }
    // read file
    int pair[3];
    for (int i = 0; i < num_edge; ++i) {
        fread(pair, sizeof(int), 3, file);
        int from = pair[0], to = pair[1], weight = pair[2];
        dist[from * padded_num_vertex + to] = weight;
        rem_orig_dist[from * padded_num_vertex + to] = weight;
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
    int in_block_j = threadIdx.x *4;
    // load gloval data into my basic block's share memory
    extern __shared__ int share_target[BLOCKED_SQUARE_SIZE][BLOCKED_SQUARE_SIZE];
    for(int id = 0; id < THREAD_LOAD_SAVE_NUM; id++){
        share_target[in_block_i][in_block_j + id] =  d_dist[(glb_block_i*BLOCKED_SQUARE_SIZE+in_block_i) * padded_num_vertex + (glb_block_j * BLOCKED_SQUARE_SIZE + in_block_j) + id];
    }
    __syncthreads();
    // start computing and enumerate k
    for(int k=0;k<BLOCKED_SQUARE_SIZE;k++){
        for(int id = 0; id<THREAD_LOAD_SAVE_NUM; id++){
            //new from in_block_i to in_block_j+id          old from in_block_i to in_block_j+id                    from  in_block_i to k, from k to in_block_j+id
            share_target[in_block_i][in_block_j + id] = min(share_target[in_block_i][in_block_j + id], share_target[in_block_i][k] + share_target[k][in_block_j + id]);
        }
        // there are dependence inbetween k at phase 1
        __syncthreads();
    }
    // write back
    for(int id = 0; id < THREAD_LOAD_SAVE_NUM; id++){
        d_dist[(glb_block_i*BLOCKED_SQUARE_SIZE+in_block_i) * padded_num_vertex + (glb_block_j * BLOCKED_SQUARE_SIZE + in_block_j) + id] = share_target[in_block_i][in_block_j + id];
    }
}
__global__ void phase_2(int* d_dist, int padded_num_vertex, int round){

}
__global__ void phase_3(int* d_dist, int padded_num_vertex, int round){

}

void blocked_floyd_warshell(bool debug, bool check_phase[4]){
    // partition the matrix into blocks
    // each block is BLOCKED_SQUARE_SIZE * BLOCKED_SQUARE_SIZE
    // each block is stored in row-major order
    // each block is stored in dist[i * padded_num_vertex + j], where i and j are the top-left corner of the block

    // num_blocked_square_row is guaranteed to be an integer, for padded_num_vertex = COALESCED_FACTOR * k, which is a multiple of BLOCKED_SQUARE_SIZE
    num_blocked_square_row = padded_num_vertex / BLOCKED_SQUARE_SIZE;
    num_blocked_square_col = padded_num_vertex / BLOCKED_SQUARE_SIZE;
    dim3 basic_block(THREAD_X_DIM, THREAD_Y_DIM);
    dim3 phase_1_grid(1);
    dim3 phase_2_grid(num_blocked_square_row);
    dim3 phase_3_grid(num_blocked_square_row, num_blocked_square_row);
    if(debug){
        // show num_vertex, num_edge, num_blocked_square_row, padded_num_vertex
        cout << "======vertex info=============================================="<< endl;
        cout << "num_vertex: " << num_vertex << " ,num_edge: " << num_edge << endl;
        cout << "num_blocked_square_row: " << num_blocked_square_row << " ,padded_num_vertex: " << padded_num_vertex << endl;
        cout << "======grid and block info======================================" << endl;
        cout << "num_blocked_square_row: " << num_blocked_square_row << endl;
        cout << "basic_block.x: " << basic_block.x << " ,basic_block.y: " << basic_block.y << endl;
        cout << "phase_1_grid.x: " << phase_1_grid.x << " ,phase_1_grid.y: " << phase_1_grid.y << endl;
        cout << "phase_2_grid.x: " << phase_2_grid.x << " ,phase_2_grid.y: " << phase_2_grid.y << endl;
        cout << "phase_3_grid.x: " << phase_3_grid.x << " ,phase_3_grid.y: " << phase_3_grid.y << endl;
        return;
    }
    
    // allocate memory for dist in device
    int* d_dist;
    cudaMalloc(&d_dist, padded_num_vertex * padded_num_vertex * sizeof(int));
    // copy dist from host to device
    cudaMemcpy(d_dist, dist, padded_num_vertex * padded_num_vertex * sizeof(int), cudaMemcpyHostToDevice);
    cout << "num_block_square_row: " << num_blocked_square_row << endl;
    for(int round = 0; round< num_blocked_square_row; round++){
        cout << "round: " << round << endl;
        // three phases:
        // phase 1: dependent block, should be thread sync at the level of k
        phase_1 <<< phase_1_grid, basic_block>>> (d_dist, padded_num_vertex, round);
        // // phase 2: dependent block, should be thread sync at the level of k 
        // if(check_phase[2]) phase_2 <<< phase_2_grid, basic_block>>> (d_dist, padded_num_vertex, round);
        // // phase 3: independent block, any place is ok
        // if(check_phase[3]) phase_3 <<< phase_3_grid, basic_block>>> (d_dist, padded_num_vertex, round);
    }
    // copy dist from device to host
    cudaMemcpy(dist, d_dist, padded_num_vertex * padded_num_vertex * sizeof(int), cudaMemcpyDeviceToHost);
}

void check(char* o_file){
    FILE* file = fopen(o_file, "rb");
    cout << "show calc. res: " << endl;
    for (int i = 0; i < num_vertex; i++) {
        for (int j = 0; j < num_vertex; j++) {
            int a;
            fread(&a, sizeof(int), 1, file);
            if (a != NOT_REACHABLE)
                cout << setw(maxWidth + 1) << a;
            else
                cout << setw(maxWidth + 1) << "x";
        }
        cout << endl;
    }
    fclose(file);
    // handle rem-orig dist
    cout << "at sample output" << endl;
    cout << "BSS: "<< BLOCKED_SQUARE_SIZE << endl;
    for(int round = 0; round< num_blocked_square_row; round++){
        cout << "round: " << round << ": ";
        // three phases:
        // phase 1: dependent block, should be thread sync at the level of k
        cout << (round*BLOCKED_SQUARE_SIZE) << " " << (round*BLOCKED_SQUARE_SIZE) << endl;
        for(int k=0;k<BLOCKED_SQUARE_SIZE;k++){
            for(int in_block_i = 0; in_block_i<BLOCKED_SQUARE_SIZE; in_block_i++){
                for(int in_block_j=0;in_block_j<BLOCKED_SQUARE_SIZE;in_block_j++){
                    int a = rem_orig_dist[(round*BLOCKED_SQUARE_SIZE+in_block_i)*padded_num_vertex + (round*BLOCKED_SQUARE_SIZE+in_block_j)];
                    int b = rem_orig_dist[(round*BLOCKED_SQUARE_SIZE+in_block_i)*padded_num_vertex + (round*BLOCKED_SQUARE_SIZE+k)];
                    int c = rem_orig_dist[(round*BLOCKED_SQUARE_SIZE+k         )*padded_num_vertex + (round*BLOCKED_SQUARE_SIZE+in_block_j)];
                    rem_orig_dist[(round*BLOCKED_SQUARE_SIZE+in_block_i)*padded_num_vertex + (round*BLOCKED_SQUARE_SIZE+in_block_j)] = min(a, b+c);
                }
            }
        }
    }
    for (int i = 0; i < num_vertex; i++) {
        for (int j = 0; j < num_vertex; j++) {
            int a = rem_orig_dist[i*padded_num_vertex + j];
            if(a!=dist[i*num_vertex+j]){
                cout << "fuck up================" << endl;
                return;
            }
            if (a != NOT_REACHABLE)
                cout << setw(maxWidth + 1) << a;
            else
                cout << setw(maxWidth + 1) << "x";
        }
        cout << endl;
    }
    cout << "sound safe~~" << endl;
}

int main(int argc, char* argv[]){
    assert(argc == 3);
    maxWidth = 5;
    bool debug_flag = false;
    bool check_phase[4];
    char* input_file = argv[1];
    char* output_file = argv[2];
    // parse input file
    input(input_file);
    // blocked floyd-warshall
    for(int i=0;i<4;i++) check_phase[i] =false;
    check_phase[1] = true;
    blocked_floyd_warshell(debug_flag, check_phase);
    // output file
    output(output_file);
    // check
    check(output_file);
    return 0;
}