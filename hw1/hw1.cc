#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <iostream>
#include <math.h>
#include <limits>

#define EVEN_PHASE 123
#define ODD_PHASE 456
#define SINGLE_ELEMENT 1

// compare float in ascending order
int compare(const void *a, const void *b){
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa > fb)
        return 1;
    if (fa < fb)
        return -1;
    return 0;
}

void distribute_portion(float* data, float* tmp, int each_hold, bool take_small_portion){
    int write_index, data_index, tmp_index;
    float* buf = new float[each_hold];
    if(take_small_portion){
        write_index = 0;
        data_index = 0;
        tmp_index = 0;
        while(write_index<each_hold){
            if(data[data_index] < tmp[tmp_index]){
                buf[write_index] = data[data_index];
                data_index++;
            }else{
                buf[write_index] = tmp[tmp_index];
                tmp_index++;
            }
            write_index++;
        }
    }else{
        write_index = each_hold-1;
        data_index = each_hold-1;
        tmp_index = each_hold-1;
        while(write_index>=0){
            if(data[data_index] > tmp[tmp_index]){
                buf[write_index] = data[data_index];
                data_index--;
            }else{
                buf[write_index] = tmp[tmp_index];
                tmp_index--;
            }
            write_index--;
        }
    }
    for(int i=0;i<each_hold;i++)
        data[i] = buf[i];
    delete[] buf;
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int total_num_float = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    
    // early stop for single process or small data
    if(size==1 || total_num_float<=size){
        float *data = new float[total_num_float];
        MPI_File input_file, output_file;
        MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_File_read_at(input_file, 0, data, total_num_float, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&input_file);
        qsort(data, total_num_float, sizeof(float), compare);
        MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
        MPI_File_write_at(output_file, 0, data, total_num_float, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&output_file);
        delete[] data;
        MPI_Finalize();
        return 0;
    }

    int each_hold, last_hold;
    each_hold = std::ceil((total_num_float+size-1)/size);
    //printf("n: %d, size:%d, rank %d each hold %d\n",total_num_float, size, rank, each_hold);
    if(total_num_float%each_hold==0) last_hold = each_hold;
    else last_hold = total_num_float%each_hold;
    
    MPI_File input_file, output_file;

    float *data = new float[each_hold];
    float *tmp = new float[each_hold];

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    if(rank!=size-1){// noraml process
        MPI_File_read_at(input_file, sizeof(float) * rank * each_hold, data, each_hold, MPI_FLOAT, MPI_STATUS_IGNORE);
    }else{// last process
        MPI_File_read_at(input_file, sizeof(float) * rank * each_hold, data, last_hold, MPI_FLOAT, MPI_STATUS_IGNORE);
        // the rest of the data should be float max
        for(int i=last_hold;i<each_hold;i++) data[i] = std::numeric_limits<float>::max();
    }
    MPI_File_close(&input_file);
    // preprocess data to be ascending order
    qsort(data, each_hold, sizeof(float), compare);
    
    // implement odd even sort
    for(int iteration = 0;iteration<size; iteration++){
        /*
        even phase. take even process as left and odd process as right: [even, odd]
            query even.data[each_hold-1] and odd.data.[0]. 
            if even.data.[each_hold-1] > odd.data.[0], apply large communication. 
                large communication means send all data to the other process.
                After large communication, we can have even.data[0:each_hold-1](our data) and even.tmp[0:each_hold-1](other process data). Total length is 2*each_hold.
                Then, we can merge them one by one by swap for they are already sorted.
                large portion for odd and small portion for even.
            else, stop communication.
        */

        if(rank%2==0){
            if(rank!=size-1){
                MPI_Sendrecv(   data+each_hold-1, SINGLE_ELEMENT, MPI_FLOAT, 
                                rank+1, EVEN_PHASE,
                                tmp, SINGLE_ELEMENT, MPI_FLOAT, 
                                rank+1, EVEN_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(data[each_hold-1] > tmp[0]){
                    // start large communication
                    MPI_Sendrecv(   data, each_hold, MPI_FLOAT, 
                                    rank+1, EVEN_PHASE,
                                    tmp, each_hold, MPI_FLOAT, 
                                    rank+1, EVEN_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    distribute_portion(data, tmp, each_hold, true);
                }
            }
        }else{
            MPI_Sendrecv(   data, SINGLE_ELEMENT, MPI_FLOAT, 
                            rank-1, EVEN_PHASE,
                            tmp+each_hold-1, SINGLE_ELEMENT, MPI_FLOAT, 
                            rank-1, EVEN_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(data[0] < tmp[each_hold-1]){
                // start large communication
                MPI_Sendrecv(   data, each_hold, MPI_FLOAT, 
                                rank-1, EVEN_PHASE,
                                tmp, each_hold, MPI_FLOAT, 
                                rank-1, EVEN_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // merge data and tmp
                distribute_portion(data, tmp, each_hold, false);
            }
        }

        // odd phase. take odd process as left and even process as right: [odd, even]
        if(rank%2==1){
            if(rank!=size-1){
                MPI_Sendrecv(   data+each_hold-1, SINGLE_ELEMENT, MPI_FLOAT, 
                                rank+1, ODD_PHASE,
                                tmp, SINGLE_ELEMENT, MPI_FLOAT, 
                                rank+1, ODD_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(data[each_hold-1] > tmp[0]){
                    // start large communication
                    MPI_Sendrecv(   data, each_hold, MPI_FLOAT, 
                                    rank+1, ODD_PHASE,
                                    tmp, each_hold, MPI_FLOAT, 
                                    rank+1, ODD_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // merge data and tmp
                    distribute_portion(data, tmp, each_hold, true);
                }
            }
        }else{
            if (rank==0) continue;
            MPI_Sendrecv(   data, SINGLE_ELEMENT, MPI_FLOAT, 
                            rank-1, ODD_PHASE,
                            tmp+each_hold-1, SINGLE_ELEMENT, MPI_FLOAT, 
                            rank-1, ODD_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(data[0] < tmp[each_hold-1]){
                // start large communication
                MPI_Sendrecv(   data, each_hold, MPI_FLOAT, 
                                rank-1, ODD_PHASE,
                                tmp, each_hold, MPI_FLOAT, 
                                rank-1, ODD_PHASE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // merge data and tmp
                distribute_portion(data, tmp, each_hold, false);
            }
        }
    }

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if(rank!=size-1){
        //printf("rank %d write %d floats\n", rank, each_hold);
        MPI_File_write_at(output_file, sizeof(float) * rank * each_hold, data, each_hold, MPI_FLOAT, MPI_STATUS_IGNORE);
    }else{
        //printf("rank %d write %d floats\n", rank, last_hold);
        MPI_File_write_at(output_file, sizeof(float) * rank * each_hold, data, last_hold, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&output_file);
    // release memory
    delete[] data;
    delete[] tmp;
    MPI_Finalize();
    return 0;
}

