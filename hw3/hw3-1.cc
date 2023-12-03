#include <sched.h>
#include <iostream>
#include <pthread.h>
#include <vector>
#include <fstream>
#include <cassert>
#include <unordered_map>
#include <climits>
#include <iomanip>
#include <cmath>
#include <queue>
#include <omp.h>

#define NO_PATH ((1 << 30) - 1) // 2^30 - 1
#define B (32)
using namespace std;

int num_vertices, num_edges, num_blocks, total_threads;
int** dist;

typedef struct block_job{
    int row, col;
} block_job;

void edge_disp(int** matrix) {
    const int field_width = 5;  // Adjust the width based on your needs

    std::cout << std::setw(field_width) << " "; // Top-left corner space

    // Print column headers
    for (int j = 0; j < num_vertices; j++) {
        std::cout << std::setw(field_width) << j;
    }
    std::cout << std::endl;

    for (int i = 0; i < num_vertices; i++) {
        // Print row header
        std::cout << std::setw(field_width) << i;

        for (int j = 0; j < num_vertices; j++) {
            std::cout << std::setw(field_width);
            
            if (matrix[i][j] == NO_PATH)
                std::cout << "NP";
            else
                std::cout << matrix[i][j];
        }
        std::cout << std::endl;
    }
}

int input_parsing(char* file_name){
    ifstream file(file_name, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Error opening file: " << file_name << std::endl;
        return 1;
    }

    // Get the size of the file
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the binary data into a vector
    std::vector<char> binary_data(file_size);
    file.read(binary_data.data(), file_size);

    // Assuming each integer is 4 bytes (32 bits), you can convert the binary data to integers
    // Note: This assumes a little-endian system; adjust if your data is big-endian
    const int* integers = reinterpret_cast<const int*>(binary_data.data());
    // take out the first integer: # vertices
    num_vertices = integers[0];
    // construct the distance matrix
    dist = new int*[num_vertices];
    for(int i=0; i<num_vertices; i++){
        dist[i] = new int[num_vertices];
        for(int j=0; j<num_vertices; j++){
            if(i==j) dist[i][j] = 0;
            else dist[i][j] = NO_PATH;
        }
    }
    num_edges = integers[1];
    for(int i=2; i<2+num_edges*3; i+=3){
        int src = integers[i];
        int dst = integers[i+1];
        int weight = integers[i+2];
        dist[src][dst] = weight;
    }
    return 0;
}

void omp_warshell(){
    cout << "omp floyd warshell" << endl;
    // copy the distance matrix
    for(int k=0; k<num_vertices; k++){
        #pragma omp parallel for schedule(guided, 16) collapse(2)
        for(int i=0; i<num_vertices; i++){
            for(int j=0; j<num_vertices; j++){
                // invalid case for INT_MAX
                if(dist[i][j] > dist[i][k]+dist[k][j] && dist[i][k]!=NO_PATH && dist[k][j]!=NO_PATH){
                    dist[i][j] = dist[i][k]+dist[k][j];
                }
            }
        }
    }
    return;
}

int** basic_floyd_warshell(){
    cout << "basic flody warshell" << endl;
    // copy the distance matrix
    int **tmp_dist = new int*[num_vertices];
    for(int i=0; i<num_vertices; i++){
        tmp_dist[i] = new int[num_vertices];
        for(int j=0; j<num_vertices; j++){
            tmp_dist[i][j] = dist[i][j];
        }
    }
    for(int k=0; k<num_vertices; k++){
        for(int i=0; i<num_vertices; i++){
            for(int j=0; j<num_vertices; j++){
                // invalid case for INT_MAX
                if(tmp_dist[i][k]==NO_PATH || tmp_dist[k][j]==NO_PATH){
                    continue;
                }else{
                    tmp_dist[i][j] = min(tmp_dist[i][j], tmp_dist[i][k]+tmp_dist[k][j]);
                }
            }
        }
    }
    return tmp_dist;
}

void show_basic_stat(){
    // display B, num_blocks, num_vertices, num_edges, NO_PATH in one line
    cout << "B: "<< B << " , num_blocks: " << num_blocks << " , num_vertices: " << num_vertices << " , num_edges: " << num_edges << " , NO_PATH: " << NO_PATH << endl;
}

void block_floyd_warshell(){
    cout << "block floyd warshell" << endl;
    for(int round = 0;round<num_blocks;round++){
        // phase 1 (primary block), handle self-dependent block
        // this block is the diagonal block, apply basic floyd warshell over this block
        int target_row = round * B, target_col = round * B;
        for (int k = target_col; k < target_col+B; k++){
            for (int i = target_row; i < target_row+B; i++){
                for (int j = target_col; j < target_col+B; j++){
                    // avoid invalid memory access for the last block
                    if(i>=num_vertices || j>=num_vertices || k>=num_vertices) continue;
                    // invalid case for INT_MAX
                    if(dist[i][k]==NO_PATH || dist[k][j]==NO_PATH) continue;
                    // update the distance
                    dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j]);
                }
            }
        }
        
        // phase 2 and 3. handle the rest of blocks
        // order: 
        // phase2 block right to primary block
        // phase2 block above to primary block
        // phase3 block that sits above and right to primary block
        // phase2 block left of primary block
        // phase3 block that sits above and left to primary block
        // phase2 block below to primary block
        // phase3 block that sits below and right to primary block
        // phase3 block that sits below and left to primary block
        // phase3 block that sits below and right to primary block

        // put them into a queue and pop them out one by one
        queue<block_job> job_queue;
        // phase2 block right to primary block
        for(int tmp_col = target_col+B; tmp_col<num_vertices; tmp_col+=B){
            block_job tmp_job;
            tmp_job.row = target_row;
            tmp_job.col = tmp_col;
            job_queue.push(tmp_job);
        }
        // phase2 block above to primary block
        for(int tmp_row = target_row-B; tmp_row>=0; tmp_row-=B){
            block_job tmp_job;
            tmp_job.row = tmp_row;
            tmp_job.col = target_col;
            job_queue.push(tmp_job);
        }
        // phase3 block that sits above and right to primary block
        for(int tmp_row = target_row-B; tmp_row>=0; tmp_row-=B){
            for(int tmp_col = target_col+B; tmp_col<num_vertices; tmp_col+=B){
                block_job tmp_job;
                tmp_job.row = tmp_row;
                tmp_job.col = tmp_col;
                job_queue.push(tmp_job);
            }
        }
        // phase2 block left of primary block
        for(int tmp_col = target_col-B; tmp_col>=0; tmp_col-=B){
            block_job tmp_job;
            tmp_job.row = target_row;
            tmp_job.col = tmp_col;
            job_queue.push(tmp_job);
        }
        // phase3 block that sits above and left to primary block
        for(int tmp_row = target_row-B; tmp_row>=0; tmp_row-=B){
            for(int tmp_col = target_col-B; tmp_col>=0; tmp_col-=B){
                block_job tmp_job;
                tmp_job.row = tmp_row;
                tmp_job.col = tmp_col;
                job_queue.push(tmp_job);
            }
        }
        // phase2 block below to primary block
        for(int tmp_row = target_row+B; tmp_row<num_vertices; tmp_row+=B){
            block_job tmp_job;
            tmp_job.row = tmp_row;
            tmp_job.col = target_col;
            job_queue.push(tmp_job);
        }
        // phase3 block that sits below and right to primary block
        for(int tmp_row = target_row+B; tmp_row<num_vertices; tmp_row+=B){
            for(int tmp_col = target_col+B; tmp_col<num_vertices; tmp_col+=B){
                block_job tmp_job;
                tmp_job.row = tmp_row;
                tmp_job.col = tmp_col;
                job_queue.push(tmp_job);
            }
        }
        // phase3 block that sits below and left to primary block
        for(int tmp_row = target_row+B; tmp_row<num_vertices; tmp_row+=B){
            for(int tmp_col = target_col-B; tmp_col>=0; tmp_col-=B){
                block_job tmp_job;
                tmp_job.row = tmp_row;
                tmp_job.col = tmp_col;
                job_queue.push(tmp_job);
            }
        }
        // phase3 block that sits below and right to primary block
        for(int tmp_row = target_row+B; tmp_row<num_vertices; tmp_row+=B){
            for(int tmp_col = target_col+B; tmp_col<num_vertices; tmp_col+=B){
                block_job tmp_job;
                tmp_job.row = tmp_row;
                tmp_job.col = tmp_col;
                job_queue.push(tmp_job);
            }
        }

        // pop out the job and do the calculation
        #pragma omp parallel
        {
            while(!job_queue.empty()){
                // get the job
                block_job tmp_job;
                #pragma omp critical
                {
                    tmp_job = job_queue.front();
                    job_queue.pop();
                }
                //// do the calculation
                #pragma omp for
                for(int k = target_col; k < target_col+B; k++){
                    for(int i = tmp_job.row; i < tmp_job.row+B; i++){
                        for(int j = tmp_job.col; j < tmp_job.col+B; j++){
                            // avoid invalid memory access for the last block
                            if(i>=num_vertices || j>=num_vertices || k>=num_vertices) continue;
                            // invalid case for INT_MAX
                            if(dist[i][k]==NO_PATH || dist[k][j]==NO_PATH) continue;
                            // update the distance
                            dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j]);
                        }
                    }
                }
                
            }
        }
    }
}

bool check_correctness(int** tmp_dist){
    for(int i=0; i<num_vertices; i++){
        for(int j=0; j<num_vertices; j++){
            if(tmp_dist[i][j]!=dist[i][j]){
                cout << "wrong at " << i << " " << j << endl;
                return false;
            }
        }
    }
    return true;
}

int write_back(char* output_file_name){
    ofstream output_file(output_file_name, std::ios::binary);
    if(!output_file.is_open()) {
        std::cerr << "Error opening file: " << output_file_name << std::endl;
        return -1;
    }
    // write the distance matrix
    // parallel output write
    for(int i=0; i<num_vertices; i++){
        for(int j=0; j<num_vertices; j++){
            output_file.write(reinterpret_cast<const char*>(&dist[i][j]), sizeof(int));
        }
    }
    output_file.close();
    return 0;
}

int main(int argc, char* argv[]){
    char* input_file_name = argv[1];
    char* output_file_name = argv[2];
    double total_time, read_time, write_time, compute_time;
    double start_time, end_time;
    // count read time
    start_time = omp_get_wtime();
    
    assert(input_parsing(input_file_name)==0);
    
    end_time = omp_get_wtime();
    read_time = end_time - start_time;

    num_blocks = static_cast<int>(std::ceil(static_cast<double>(num_vertices) / B));
    //block_floyd_warshell();
    // count the time
    start_time = omp_get_wtime();
    
    omp_warshell();
    
    end_time = omp_get_wtime();
    compute_time = end_time - start_time;
    // count write time
    start_time = omp_get_wtime();
    
    assert(write_back(output_file_name)==0);
    
    end_time = omp_get_wtime();
    write_time = end_time - start_time;
    total_time = read_time + compute_time + write_time;
    cout << "total time: " << total_time << endl;
    cout << "read time: " << read_time << " compute time: " << compute_time << " write time: " << write_time << endl;
    return 0;
}