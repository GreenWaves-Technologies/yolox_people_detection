#include "camera.h"


void shift_bits(unsigned char *cam_image, int height, int width){

    for (int i = 0; i < 2*height; i++) { // Put pixels on 8 bits instead of 10 to go on 1 byte encoding only
        for (int j = 0; j < width; j++) {
            // Shifts bits to delete the 2 LSB, on the 10 useful bits
            cam_image[i * width + j] = (cam_image[(i * width + j) *2 +1] << 6) | (cam_image[(i * width + j) *2] >> 2);
        }
    }

}