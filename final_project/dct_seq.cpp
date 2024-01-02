#include <stdio.h>
#include <jpeglib.h>
#include <png.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#define FORWARD true
#define BACKWARD false
using namespace std;

double DCT_MATRIX[8][8] = {
    {0.3535534, 0.3535534, 0.3535534, 0.3535534, 0.3535534, 0.3535534, 0.3535534, 0.3535534},
    {0.4903926, 0.4157348, 0.2777851, 0.0975452, -0.0975452, -0.2777851, -0.4157348, -0.4903926},
    {0.4619398, 0.1913417, -0.1913417, -0.4619398, -0.4619398, -0.1913417, 0.1913417, 0.4619398},
    {0.4157348, -0.0975452, -0.4903926, -0.2777851, 0.2777851, 0.4903926, 0.0975452, -0.4157348},
    {0.3535534, -0.3535534, -0.3535534, 0.3535534, 0.3535534, -0.3535534, -0.3535534, 0.3535534},
    {0.2777851, -0.4903926, 0.0975452, 0.4157348, -0.4157348, -0.0975452, 0.4903926, -0.2777851},
    {0.1913417, -0.4619398, 0.4619398, -0.1913417, -0.1913417, 0.4619398, -0.4619398, 0.1913417},
    {0.0975452, -0.2777851, 0.4157348, -0.4903926, 0.4903926, -0.4157348, 0.2777851, -0.0975452}
};

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

int width, height, components;

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

unsigned char*** read_jpg(){
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;

    if ((infile = fopen("./src/Barbara.jpg", "rb")) == NULL) {
        fprintf(stderr, "can't open %s\n", "./src/Barbara.jpg");
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

    // Allocate memory for the 3D array
    unsigned char*** image = new unsigned char**[cinfo.output_height];
    for (unsigned int y = 0; y < cinfo.output_height; y++) {
        image[y] = new unsigned char*[cinfo.output_width];
        for (unsigned int x = 0; x < cinfo.output_width; x++) {
            image[y][x] = new unsigned char[3];
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

    cout << "Image data: " << endl;
    cout << "height: " << cinfo.output_height << endl;
    cout << "width: " << cinfo.output_width << endl;
    cout << "components: " << cinfo.output_components << endl;

    width = cinfo.output_width;
    height = cinfo.output_height;
    components = cinfo.output_components;

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

void center_data(double*** image, bool direction){
    if(direction){
        // center the data
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                for(int c=0;c<components;c++){
                    image[i][j][c] -= 128;
                }
            }
        }
    }else{
        // decenter the data
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                for(int c=0;c<components;c++){
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

void generate_dct_matrix() {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (i == 0) {
                DCT_MATRIX[i][j] = 1 / std::sqrt(8);
            } else {
                DCT_MATRIX[i][j] = std::sqrt(2.0 / 8) * std::cos(((2 * j + 1) * i * M_PI) / 16);
            }
        }
    }
}

double*** dct_compression(double*** image){
    double*** res = allocate_3d_double_array(height, width, 3);
    // copy the image data
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            for(int c=0;c<components;c++){
                res[i][j][c] = image[i][j][c];
            }
        }
    }
    //center_data(res, FORWARD);
    // cut the image into 8x8 blocks
    for(int i=0;i<height;i+=8){
        for(int j=0;j<width;j+=8){
            for(int c=0;c<components;c++){
                // apply dct
                double temp[8][8];
                double temp_dct[8][8];
                double temp_idct[8][8];
                // copy the data into temp
                for(int ii=0;ii<8;ii++){
                    for(int jj=0;jj<8;jj++){
                        temp[ii][jj] = res[i+ii][j+jj][c];
                    }
                }
                // center tmp
                center_tmp(temp, FORWARD);
                // apply dct and store the result in temp_dct
                for(int ii=0;ii<8;ii++){
                    for(int jj=0;jj<8;jj++){
                        double sum = 0;
                        for(int k=0;k<8;k++){
                            sum += DCT_MATRIX[ii][k] * temp[k][jj];
                        }
                        temp_dct[ii][jj] = sum;
                    }
                }
                // apply idct and store the result in temp_idct
                for(int ii=0;ii<8;ii++){
                    for(int jj=0;jj<8;jj++){
                        double sum = 0;
                        for(int k=0;k<8;k++){
                            sum += DCT_MATRIX[k][ii] * temp_dct[k][jj];
                        }
                        temp_idct[ii][jj] = sum;
                    }
                }
                // decenter tmp
                center_tmp(temp_idct, BACKWARD);
                // copy the data back to res
                for(int ii=0;ii<8;ii++){
                    for(int jj=0;jj<8;jj++){
                        res[i+ii][j+jj][c] = temp_idct[ii][jj];
                    }
                }
            }
        }
    }
    //center_data(res, BACKWARD);
    return res;
}

int main() {
    // Read the image data into a 3D array
    unsigned char*** image = read_jpg();
    // Process the image data here
    double*** image_YCBCR = RGB2YCBCR(image);
    generate_dct_matrix();
    // // display the dct matrix
    // cout << "DCT Matrix: " << endl;
    // for(int i=0;i<8;i++){
    //     for(int j=0;j<8;j++){
    //         cout << DCT_MATRIX[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    double*** image_dct = dct_compression(image_YCBCR);
    unsigned char*** image_RGB_orig = YCBCR2RGB(image_YCBCR);
    unsigned char*** image_RGB_dct = YCBCR2RGB(image_dct);

    // Write the image data to a PNG file
    write_png("./out/dct_seq_Barbara_orig.png", image_RGB_orig, width, height);
    write_png("./out/dct_seq_Barbara_dct.png", image_RGB_dct, width, height);
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

    return 0;
}
