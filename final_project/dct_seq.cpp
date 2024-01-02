#include <stdio.h>
#include <jpeglib.h>
#include <png.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

int width, height, components;

// Write a PNG file from a 3D array of RGB values
void write_png(const char* filename, unsigned char*** image_data, int width, int height) {
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
            row[x*3] = image_data[y][x][0];   // Red
            row[x*3 + 1] = image_data[y][x][1]; // Green
            row[x*3 + 2] = image_data[y][x][2]; // Blue
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
    unsigned char*** image_data = new unsigned char**[cinfo.output_height];
    for (unsigned int y = 0; y < cinfo.output_height; y++) {
        image_data[y] = new unsigned char*[cinfo.output_width];
        for (unsigned int x = 0; x < cinfo.output_width; x++) {
            image_data[y][x] = new unsigned char[3];
        }
    }

    unsigned int current_row = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (unsigned int x = 0; x < cinfo.output_width; x++) {
            image_data[current_row][x][0] = buffer[0][x * 3];     // Red
            image_data[current_row][x][1] = buffer[0][x * 3 + 1]; // Green
            image_data[current_row][x][2] = buffer[0][x * 3 + 2]; // Blue
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

    return image_data;
}



int main() {
    // Read the image data into a 3D array
    unsigned char*** image_data = read_jpg();
    // Process the image data here
    

    // Write the image data to a PNG file
    write_png("./out/dct_seq_Barbara.png", image_data, width, height);
    // Free the 3D array
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            delete[] image_data[y][x];
        }
        delete[] image_data[y];
    }
    delete[] image_data;

    return 0;
}
