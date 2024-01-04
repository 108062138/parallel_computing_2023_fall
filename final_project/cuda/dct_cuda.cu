#include <stdio.h>
#include <jpeglib.h>
#include <png.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <omp.h>

#define FORWARD true
#define BACKWARD false
#define MOVE_DATA_FROM_IMAGE_TO_TMP true
#define MOVE_DATA_FROM_TMP_TO_IMAGE false

#define BLOCK_X_SIZE 8
#define BLOCK_Y_SIZE 8

using namespace std;

__constant__ double d_DCT_MATRIX[8][8];
__constant__ double d_DCT_MATRIX_T[8][8];
__constant__ double d_LUMINANCE_TABLE[8][8];
__constant__ double d_CHROMINANCE_TABLE[8][8];

double DCT_MATRIX[8][8], DCT_MATRIX_T[8][8];

double LUMINANCE_TABLE[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 36, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

double CHROMINANCE_TABLE[8][8] = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
};

unsigned int width, height, components;
unsigned int num_blocked_square_rows, num_blocked_square_cols;

// Write a PNG file from a 3D array of RGB values
void write_png(const char* filename, unsigned char*** image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if(!fp) abort();

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    // Output is 8bit depth, RGB format
    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // Write image data
    for(int y = 0; y < height; y++) {
        png_bytep row = (png_bytep) malloc(3 * width * sizeof(png_byte));
        for (int x = 0; x < width; x++) {
            row[x*3] = image[y][x][0];   // Red
            row[x*3 + 1] = image[y][x][1]; // Green
            row[x*3 + 2] = image[y][x][2]; // Blue
        }
        png_write_row(png, row);
        free(row);
    }

    // End write
    png_write_end(png, NULL);

    if (png && info)
        png_destroy_write_struct(&png, &info);
    if (fp)
        fclose(fp);
}

unsigned char*** read_jpg(char* file_name){
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;

    if ((infile = fopen(file_name, "rb")) == NULL) {
        fprintf(stderr, "can't open %s\n", file_name);
        return 0;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    row_stride = cinfo.output_width * cinfo.output_components;

    buffer = (*cinfo.mem->alloc_sarray)
        ((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    cout << "Image data: " << endl;
    cout << "height: " << cinfo.output_height << endl;
    cout << "width: " << cinfo.output_width << endl;
    cout << "components: " << cinfo.output_components << endl;

    width = cinfo.output_width;
    height = cinfo.output_height;
    components = cinfo.output_components;
    // padding the image to be multiple of 8
    if(width%8!=0){
        width = width + 8 - width%8;
    }
    if(height%8!=0){
        height = height + 8 - height%8;
    }
    // Allocate memory for the 3D array
    unsigned char*** image = new unsigned char**[height];
    for (unsigned int y = 0; y < height; y++) {
        image[y] = new unsigned char*[width];
        for (unsigned int x = 0; x < width; x++) {
            image[y][x] = new unsigned char[3];
        }
    }

    // init the image to be all 0
    for(unsigned int i=0;i<height;i++){
        for(unsigned int j=0;j<width;j++){
            for(unsigned int c=0;c<components;c++){
                image[i][j][c] = 0;
            }
        }
    }

    unsigned int current_row = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (unsigned int x = 0; x < cinfo.output_width; x++) {
            image[current_row][x][0] = buffer[0][x * 3];     // Red
            image[current_row][x][1] = buffer[0][x * 3 + 1]; // Green
            image[current_row][x][2] = buffer[0][x * 3 + 2]; // Blue
        }
        current_row++;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return image;
}

double*** allocate_3d_double_array(unsigned int height, unsigned int width, unsigned int depth){
    double*** image = new double**[height];
    for (unsigned int y = 0; y < height; y++) {
        image[y] = new double*[width];
        for (unsigned int x = 0; x < width; x++) {
            image[y][x] = new double[depth];
        }
    }
    return image;
}

double*** RGB2YCBCR(unsigned char ***image){
    double*** image_YCBCR = allocate_3d_double_array(height, width, 3);

    // apply sub sampling
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            image_YCBCR[y][x][0] = 0.299 * image[y][x][0] + 0.587 * image[y][x][1] + 0.114 * image[y][x][2];
            image_YCBCR[y][x][1] = 128 - 0.168736 * image[y][x][0] - 0.331264 * image[y][x][1] + 0.5 * image[y][x][2];
            image_YCBCR[y][x][2] = 128 + 0.5 * image[y][x][0] - 0.418688 * image[y][x][1] - 0.081312 * image[y][x][2];
        }
    }

    return image_YCBCR;
}

unsigned char *** YCBCR2RGB(double ***image_YCBCR){
    unsigned char*** image = new unsigned char**[height];
    for (unsigned int y = 0; y < height; y++) {
        image[y] = new unsigned char*[width];
        for (unsigned int x = 0; x < width; x++) {
            image[y][x] = new unsigned char[3];
        }
    }

    // apply sub sampling
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            image[y][x][0] = image_YCBCR[y][x][0] + 1.402 * (image_YCBCR[y][x][2] - 128);
            image[y][x][1] = image_YCBCR[y][x][0] - 0.344136 * (image_YCBCR[y][x][1] - 128) - 0.714136 * (image_YCBCR[y][x][2] - 128);
            image[y][x][2] = image_YCBCR[y][x][0] + 1.772 * (image_YCBCR[y][x][1] - 128);
        }
    }

    return image;
}


std::pair<std::vector<std::vector<double>>, std::vector<double>> uniform_quantization(double block[8][8], int n, int m, int channel) {
    std::vector<std::vector<double>> res(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (channel == 0) {
                res[i][j] = block[i][j] / LUMINANCE_TABLE[i][j];
            } else {
                res[i][j] = block[i][j] / CHROMINANCE_TABLE[i][j];
            }
        }
    }
    double big = -100000000;
    double small = 100000000;
    // collect big and small
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(res[i][j] > big){
                big = res[i][j];
            }
            if(res[i][j] < small){
                small = res[i][j];
            }
        }
    }
    double total_interval = big - small;
    int step = std::pow(2, m);
    double interval_unit = total_interval / step;
    std::vector<double> ladder(step, 0);
    for (int i = 0; i < step; ++i) {
        ladder[i] = small + i * interval_unit;
    }
    auto min_element_it = std::min_element(ladder.begin(), ladder.end(), [](double a, double b) { return std::abs(a) < std::abs(b); });
    *min_element_it = 0;
    std::vector<std::vector<double>> map_on_ladder(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            auto min_element_it = std::min_element(ladder.begin(), ladder.end(), [res, i, j](double a, double b) { return std::abs(res[i][j] - a) < std::abs(res[i][j] - b); });
            map_on_ladder[i][j] = std::distance(ladder.begin(), min_element_it);
        }
    }
    return {map_on_ladder, ladder};
}

void uniform_dequantization(std::vector<std::vector<double>>& block, int n, std::vector<double>& ladder, int channel, double res[8][8]) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            block[i][j] = ladder[static_cast<int>(block[i][j])];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (channel == 0) {
                res[i][j] = std::round(block[i][j] * LUMINANCE_TABLE[i][j]);
            } else {
                res[i][j] = std::round(block[i][j] * CHROMINANCE_TABLE[i][j]);
            }
        }
    }
}

void center_data(double*** image, bool direction){
    if(direction){
        // center the data
        for(unsigned  i=0;i<height;i++){
            for(unsigned j=0;j<width;j++){
                for(unsigned int c=0;c<components;c++){
                    image[i][j][c] -= 128;
                }
            }
        }
    }else{
        // decenter the data
        for(unsigned int i=0;i<height;i++){
            for(unsigned int j=0;j<width;j++){
                for(unsigned int c=0;c<components;c++){
                    image[i][j][c] += 128;
                }
            }
        }
    }
}

void center_tmp(double tmp[8][8], bool direction){
    if(direction){
        // center the data
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++){
                tmp[i][j] -= 128;
            }
        }
    }else{
        // decenter the data
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++){
                tmp[i][j] += 128;
            }
        }
    }
}

void display_dct_matrix(){
    cout << "DCT Matrix: " << endl;
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            cout << DCT_MATRIX[i][j] << " ";
        }
        cout << endl;
    }
}

void generate_dct_matrix(bool show_dct_matrix) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (i == 0) {
                DCT_MATRIX[i][j] = 1 / std::sqrt(8);
            } else {
                DCT_MATRIX[i][j] = std::sqrt(2.0 / 8) * std::cos(((2 * j + 1) * i * M_PI) / 16);
            }
        }
    }
    // transpose
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            DCT_MATRIX_T[i][j] = DCT_MATRIX[j][i];
        }
    }
    if (show_dct_matrix)
        display_dct_matrix();
}

void matrix_mul(double A[8][8], double B[8][8], double C[8][8]){
    // a@b=c, c is output
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            double sum = 0;
            for(int k=0;k<8;k++){
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void move_data(double tmp[8][8], double*** image, int i, int j, int c, bool direction){
    if(direction==MOVE_DATA_FROM_IMAGE_TO_TMP){
        for(int ii=0;ii<8;ii++){
            for(int jj=0;jj<8;jj++){
                tmp[ii][jj] = image[i+ii][j+jj][c];
            }
        }
    }else{
        for(int ii=0;ii<8;ii++){
            for(int jj=0;jj<8;jj++){
                image[i+ii][j+jj][c] = tmp[ii][jj];
            }
        }
    }
}

void copy_image(double*** image, double*** image_copy){
    for(unsigned int i=0;i<height;i++){
        for(unsigned int j=0;j<width;j++){
            for(unsigned int c=0;c<components;c++){
                image_copy[i][j][c] = image[i][j][c];
            }
        }
    }
}

__global__ void dct_kernel(double* channel, int height, int width, int n, int m, int c){
    extern __shared__ double temp[BLOCK_Y_SIZE][BLOCK_X_SIZE];
    extern __shared__ double temp_dct_1[BLOCK_Y_SIZE][BLOCK_X_SIZE];
    extern __shared__ double temp_dct_2[BLOCK_Y_SIZE][BLOCK_X_SIZE];
    extern __shared__ double temp_idct_1[BLOCK_Y_SIZE][BLOCK_X_SIZE];
    extern __shared__ double temp_idct_2[BLOCK_Y_SIZE][BLOCK_X_SIZE];
    extern __shared__ double ladder[1024];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // load data from global memory to shared memory
    temp[threadIdx.y][threadIdx.x] = channel[i*width+j];
    __syncthreads();
    double sum;
    sum = 0;
    for(int k=0;k<8;k++)
        sum += d_DCT_MATRIX[threadIdx.y][k] * temp[k][threadIdx.x];
    temp_dct_1[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    sum = 0;
    for(int k=0;k<8;k++)
        sum += temp_dct_1[threadIdx.y][k] * d_DCT_MATRIX_T[k][threadIdx.x];
    temp_dct_2[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if(c==0)
        temp_dct_2[threadIdx.y][threadIdx.x] = temp_dct_2[threadIdx.y][threadIdx.x] / d_LUMINANCE_TABLE[threadIdx.y][threadIdx.x];
    else
        temp_dct_2[threadIdx.y][threadIdx.x] = temp_dct_2[threadIdx.y][threadIdx.x] / d_CHROMINANCE_TABLE[threadIdx.y][threadIdx.x];
    __syncthreads();
    // quantization
    // first get the min and max among temp_dct_2
    double big = -100000000;
    double small = 100000000;
    int step = 2 << m;
    for(int ii=0;ii<8;ii++){
        for(int jj=0;jj<8;jj++){
            if(temp_dct_2[ii][jj] > big){
                big = temp_dct_2[ii][jj];
            }
            if(temp_dct_2[ii][jj] < small){
                small = temp_dct_2[ii][jj];
            }
        }
    }
    double total_interval = big - small;
    double interval_unit = total_interval / step;
    for(int ii=0;ii<step;ii++){
        ladder[ii] = small + ii * interval_unit;
    }
    int closest_to_0_index = 0, closest_to_0 = 100000000;
    for(int ii=0;ii<step;ii++){
        if(std::abs(ladder[ii]) < closest_to_0){
            closest_to_0 = std::abs(ladder[ii]);
            closest_to_0_index = ii;
        }
    }
    ladder[closest_to_0_index] = 0;
    __syncthreads();
    // iterate through the ladder to find the closest value over i, j
    int closest_to_ij_index = 0, closest_to_ij = 100000000;
    for(int lad=0;lad<step;lad++){
        if(std::abs(temp_dct_2[threadIdx.y][threadIdx.x] - ladder[lad]) < std::abs(temp_dct_2[threadIdx.y][threadIdx.x] - ladder[closest_to_ij_index])){
            closest_to_ij_index = lad;
        }
    }
    temp_dct_2[threadIdx.y][threadIdx.x] = closest_to_ij_index;
    // dequantization
    temp_dct_2[threadIdx.y][threadIdx.x] = ladder[(int)temp_dct_2[threadIdx.y][threadIdx.x]];
// 
    __syncthreads();
    if(c==0)
        temp_dct_2[threadIdx.y][threadIdx.x] = temp_dct_2[threadIdx.y][threadIdx.x] * d_LUMINANCE_TABLE[threadIdx.y][threadIdx.x];
    else
        temp_dct_2[threadIdx.y][threadIdx.x] = temp_dct_2[threadIdx.y][threadIdx.x] * d_CHROMINANCE_TABLE[threadIdx.y][threadIdx.x];
    __syncthreads();
    // inverse DCT
    sum = 0;
    for(int k=0;k<8;k++)
        sum += d_DCT_MATRIX_T[threadIdx.y][k] * temp_dct_2[k][threadIdx.x];
    temp_idct_1[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    sum = 0;
    for(int k=0;k<8;k++)
        sum += temp_idct_1[threadIdx.y][k] * d_DCT_MATRIX[k][threadIdx.x];
    temp_idct_2[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();

    // write data back to global memory
    channel[i*width+j] = temp_idct_2[threadIdx.y][threadIdx.x];
}

double*** dct_compression(double*** image){
    int m=8, n=5; // m: quantization level, n: store subblock size
    bool show_dct_matrix = false;
    // generate dct matrix
    generate_dct_matrix(show_dct_matrix);
    double*** res = allocate_3d_double_array(height, width, 3);
    copy_image(image, res);
    // cut the image into 8x8 blocks
    center_data(res, FORWARD);

    // create three channels to store each channel's dct result
    double* channels[components];
    double* d_channels[components];
    num_blocked_square_rows = height / BLOCK_Y_SIZE;
    num_blocked_square_cols = width / BLOCK_X_SIZE;

    dim3 block(BLOCK_X_SIZE, BLOCK_Y_SIZE);
    dim3 grid(num_blocked_square_cols, num_blocked_square_rows);

    // put DCT_MATRIX_T into constant memory and DCT_MATRIX_T into constant memory
    cudaMemcpyToSymbol(d_DCT_MATRIX, DCT_MATRIX, 8*8*sizeof(double));
    cudaMemcpyToSymbol(d_DCT_MATRIX_T, DCT_MATRIX_T, 8*8*sizeof(double));
    cudaMemcpyToSymbol(d_LUMINANCE_TABLE, LUMINANCE_TABLE, 8*8*sizeof(double));
    cudaMemcpyToSymbol(d_CHROMINANCE_TABLE, CHROMINANCE_TABLE, 8*8*sizeof(double));
    
    for(unsigned int c=0;c<components;c++){
        channels[c] = new double[height*width];
        // copy the data into channels
        for(unsigned int i=0;i<height;i++){
            for(unsigned int j=0;j<width;j++){
                channels[c][i*width+j] = res[i][j][c];
            }
        }
        cudaMalloc((void**)&d_channels[c], height*width*sizeof(double));
        cudaMemcpy(d_channels[c], channels[c], height*width*sizeof(double), cudaMemcpyHostToDevice);
    }
    // for each channel, launch the kernel
    for(unsigned int c=0;c<components;c++){
        dct_kernel<<<grid, block>>>(d_channels[c], height, width, n, m, c);
    }
    // copy the data back to channels
    for(unsigned int c=0;c<components;c++){
        cudaMemcpy(channels[c], d_channels[c], height*width*sizeof(double), cudaMemcpyDeviceToHost);
    }
    // copy the data back to res
    for(unsigned int i=0;i<height;i++){
        for(unsigned int j=0;j<width;j++){
            for(unsigned int c=0;c<components;c++){
                res[i][j][c] = channels[c][i*width+j];
            }
        }
    }
    center_data(res, BACKWARD);
    return res;
}

int main(int argc, char* argv[]) {
    // read the command line arguments
    for(int i=0;i<argc;i++)
        cout << argv[i] << endl;
    char *src_image = argv[1], *from_image = argv[2], *to_image = argv[3];

    double start_time, end_time;
    start_time = omp_get_wtime();

    // Read the image data into a 3D array
    unsigned char*** image = read_jpg(src_image);
    // Process the image data here: convert to YCbCr, DCT, quantize, inverse DCT, YCbCr to RGB
    double*** image_YCBCR = RGB2YCBCR(image);
    double*** image_dct = dct_compression(image_YCBCR);
    unsigned char*** image_RGB_orig = YCBCR2RGB(image_YCBCR);
    unsigned char*** image_RGB_dct = YCBCR2RGB(image_dct);

    // Write the image data to a PNG file
    write_png(from_image, image_RGB_orig, width, height);
    write_png(to_image, image_RGB_dct, width, height);
    // Free all 3D arrays here
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            delete[] image[y][x];
            delete[] image_YCBCR[y][x];
            delete[] image_dct[y][x];
            delete[] image_RGB_orig[y][x];
            delete[] image_RGB_dct[y][x];
        }
        delete[] image[y];
        delete[] image_YCBCR[y];
        delete[] image_dct[y];
        delete[] image_RGB_orig[y];
        delete[] image_RGB_dct[y];
    }
    delete[] image;

    end_time = omp_get_wtime();
    cout << "In omp, time: " << end_time - start_time << endl;

    return 0;
}