#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cuda_fp16.h>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8
#define BLOCK_X_DIM 512
#define BLOCK_Y_DIM 3
#define TILE_X_DIM (2+BLOCK_X_DIM+2)
#define TILE_Y_DIM 1

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {
    extern __shared__ unsigned char shared[];
    int x = blockIdx.x * blockDim.x + threadIdx.x; // true position
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(!bound_check(x, 0, width) || !bound_check(y, 0, height)) return;

    // start load shared memory
    // special case for first thread and last thread
    if(threadIdx.x == 0){
        #pragma unroll 5
        for(int i=0;i < Y;i++){
            if(bound_check(y+i - yBound, 0, height)){
                // two type of edge for x = 0
                if(bound_check(x-1, 0, width)){
                    shared[(TILE_X_DIM * i + threadIdx.x + 1) * channels + 2] = s[ (width * (y + i - yBound) + x - 1) * channels + 2];
                    shared[(TILE_X_DIM * i + threadIdx.x + 1) * channels + 1] = s[ (width * (y + i - yBound) + x - 1) * channels + 1];
                    shared[(TILE_X_DIM * i + threadIdx.x + 1) * channels + 0] = s[ (width * (y + i - yBound) + x - 1) * channels + 0];
                }
                if(bound_check(x-2, 0, width)){
                    shared[(TILE_X_DIM * i + threadIdx.x + 0) * channels + 2] = s[ (width * (y + i - yBound) + x - 2) * channels + 2];
                    shared[(TILE_X_DIM * i + threadIdx.x + 0) * channels + 1] = s[ (width * (y + i - yBound) + x - 2) * channels + 1];
                    shared[(TILE_X_DIM * i + threadIdx.x + 0) * channels + 0] = s[ (width * (y + i - yBound) + x - 2) * channels + 0];
                }
            }
        }
    }else if(threadIdx.x == BLOCK_X_DIM -1){
        #pragma unroll 5
        for(int i=0;i < Y;i++){
            if(bound_check(y+i - yBound, 0, height)){
                // two type of edge for x = 0
                if(bound_check(x+1, 0, width)){
                    shared[(TILE_X_DIM * i + threadIdx.x + 3) * channels + 2] = s[ (width * (y + i - yBound) + x + 1) * channels + 2];
                    shared[(TILE_X_DIM * i + threadIdx.x + 3) * channels + 1] = s[ (width * (y + i - yBound) + x + 1) * channels + 1];
                    shared[(TILE_X_DIM * i + threadIdx.x + 3) * channels + 0] = s[ (width * (y + i - yBound) + x + 1) * channels + 0];
                }
                if(bound_check(x+2, 0, width)){
                    shared[(TILE_X_DIM * i + threadIdx.x + 4) * channels + 2] = s[ (width * (y + i - yBound) + x + 2) * channels + 2];
                    shared[(TILE_X_DIM * i + threadIdx.x + 4) * channels + 1] = s[ (width * (y + i - yBound) + x + 2) * channels + 1];
                    shared[(TILE_X_DIM * i + threadIdx.x + 4) * channels + 0] = s[ (width * (y + i - yBound) + x + 2) * channels + 0];
                }
            }
        }
    }
    // normal load over its column (1xY)
    #pragma unroll 5
    for(int i=0;i<Y;i++){
        if(bound_check(y+i - yBound, 0, height)){
            shared[channels * (TILE_X_DIM * i + threadIdx.x + 2) + 2] = s[ (width * (y + i - yBound) + x) * channels + 2];
            shared[channels * (TILE_X_DIM * i + threadIdx.x + 2) + 1] = s[ (width * (y + i - yBound) + x) * channels + 1];
            shared[channels * (TILE_X_DIM * i + threadIdx.x + 2) + 0] = s[ (width * (y + i - yBound) + x) * channels + 0];
        }
    }
    __syncthreads();

    //computing
    /* Z axis of mask */
    float val[Z][3];
    int level = 
    for (int i = 0; i < Z; ++i) {
        val[i][2] = 0.f;
        val[i][1] = 0.f;
        val[i][0] = 0.f;
        /* Y and X axis of mask */
        for (int v = -yBound; v <= yBound; ++v) {
            for (int u = -xBound; u <= xBound; ++u) {
                if (bound_check(x + u, 0, width) && bound_check(y + v, 0, height)) {
                    const unsigned char R = shared[channels * (TILE_X_DIM * (v + yBound) + (threadIdx.x + 2 + u)) + 2];
                    const unsigned char G = shared[channels * (TILE_X_DIM * (v + yBound) + (threadIdx.x + 2 + u)) + 1];
                    const unsigned char B = shared[channels * (TILE_X_DIM * (v + yBound) + (threadIdx.x + 2 + u)) + 0];
                    val[i][2] += R * mask[i][u + xBound][v + yBound];
                    val[i][1] += G * mask[i][u + xBound][v + yBound];
                    val[i][0] += B * mask[i][u + xBound][v + yBound];
                }
            }
        }
    }
    float totalR = 0.f;
    float totalG = 0.f;
    float totalB = 0.f;
    for (int i = 0; i < Z; ++i) {
        totalR += val[i][2] * val[i][2];
        totalG += val[i][1] * val[i][1];
        totalB += val[i][0] * val[i][0];
    }
    totalR = sqrtf(totalR) / SCALE;
    totalG = sqrtf(totalG) / SCALE;
    totalB = sqrtf(totalB) / SCALE;
    const unsigned char cR = (totalR > 255.) ? 255 : totalR;
    const unsigned char cG = (totalG > 255.) ? 255 : totalG;
    const unsigned char cB = (totalB > 255.) ? 255 : totalB;
    t[channels * (width * y + x) + 2] = cR;
    t[channels * (width * y + x) + 1] = cG;
    t[channels * (width * y + x) + 0] = cB;
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // decide to use how many blocks and threads
    dim3 block_size(BLOCK_X_DIM, BLOCK_Y_DIM);
    dim3 grid_size((width + BLOCK_X_DIM -1) / BLOCK_X_DIM, (height + BLOCK_Y_DIM - 1) / BLOCK_Y_DIM);
    // launch cuda kernel
    sobel << <grid_size, block_size, TILE_X_DIM * (Y + BLOCK_Y_DIM - 1) * channels>>> (dsrc, ddst, height, width, channels);

    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}