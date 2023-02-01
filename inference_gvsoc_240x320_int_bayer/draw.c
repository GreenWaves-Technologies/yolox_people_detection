#include "draw.h"
#include "main.h"
#include "gaplib/ImgIO.h"
#include "jpeg_compress.h"

void draw_rectangle(
    unsigned char *Img, 
    int W, 
    int H, 
    int x, 
    int y, 
    int w, 
    int h, 
    unsigned char ColorValue
    ){

    int x0, x1, y0, y1;

    y0 = MAX(MIN(y, H - 1), 0);
    y1 = MAX(MIN(y + h - 1, H - 1), 0);

    x0 = x;
    y1 = h - 1;
    if (x0 >= 0 && x0 < W) {
        for (int i = y0; i <= y1; i++)
            for(int c = 0; c < CHANNELS; c++)
                Img[CHANNELS*(i * W + x0) + c] = ColorValue;
    }

    x1 = w - 1;
    if (x1 >= 0 && x1 < W) {
        for (int i = y0; i <= y1; i++)
            for(int c = 0; c < CHANNELS; c++)
                Img[CHANNELS*(i * W + x1) + c] = ColorValue;
    }

    x0 = MAX(MIN(x, W - 1), 0);
    x1 = w - 1;

    y0 = y;
    if (y0 >= 0 && y0 < H) {
        for (int i = x0; i <= x1; i++)
            for(int c = 0; c < CHANNELS; c++)
                Img[CHANNELS*(y0 * W + i) + c] = ColorValue;
    }

    if (y1 >= 0 && y1 < H) {
        for (int i = x0; i <= x1; i++)
            for(int c = 0; c < CHANNELS; c++)
                Img[CHANNELS*(y1 * W + i) + c] = ColorValue;
    }
}


void draw_boxes(
    float * model_L2_Memory_Dyn_casted,
    float * Output_1,
    int final_valid_boxes
    ){

    printf("Pointer ot L2 mem in draw_boxes: %p \n", model_L2_Memory_Dyn_casted);
    unsigned char * image = (unsigned char *) model_L2_Memory_Dyn_casted;
    printf("Casted pointer to L2 mem in draw_boxes: %p \n", image);

    int status = ReadImageFromFile(
        STR(INPUT_FILE_NAME),
        W_INP, 
        H_INP, 
        CHANNELS, 
        image,
        W_INP * H_INP * CHANNELS * sizeof(char), 
        IMGIO_OUTPUT_CHAR,
        0 
    );


    for (int i=0; i < final_valid_boxes; i++){

        int x1 = (int) Output_1[i*7 + 0];
        int y1 = (int) Output_1[i*7 + 1];
        int x2 = (int) Output_1[i*7 + 2];
        int y2 = (int) Output_1[i*7 + 3];
    

        float score = Output_1[i*7 + 4] * Output_1[i*7 + 5];
        int cls = (int) Output_1[i*7 + 6];

        draw_rectangle(image, W_INP, H_INP, x1, y1, x2, y2, 255);
    }

    /* ----------------------- SAVE IMAGE --------------------- */
    // status = WriteImageToFile(
    //     STR(OUTPUT_FILE_NAME),
    //     W_INP, 
    //     H_INP, 
    //     CHANNELS, 
    //     image,
    //     RGB888_IO // GRAY_SCALE_IO
    // ); 

    /* ----------------------- JPEG COMPRESSION --------------------- */
    printf("\n\t\t*** JPEG COMPRESSION ***\n");
    uint8_t * jpeg_img = (uint8_t *) image;
    printf("Casted to uint8_t pointer to L2 mem in draw_boxes: %p \n", jpeg_img);
    // if !(compress(image)){
        // printf("JPEG compression failed\n");
    // }

    // if compression is not successful print error message
    if (compress(jpeg_img) != 1){
        printf("JPEG compression failed\n");
    }



}





