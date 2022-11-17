#include "draw.h"
#include "model.h"
#include "gaplib/ImgIO.h"
#include "Gap.h"

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

    y1 = y + h - 1;
    if (y1 >= 0 && y1 < H) {
        for (int i = x0; i <= x1; i++)
            for(int c = 0; c < CHANNELS; c++)
                Img[CHANNELS*(y1 * W + i) + c] = ColorValue;
    }
}


void draw_boxes(
    f16 * model_L2_Memory_Dyn_casted,
    f16 * Output_1,
    int final_valid_boxes
    ){

    unsigned char * image = (unsigned char *) model_L2_Memory_Dyn_casted;
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

        int h = y2 - y1;
        int w = x2 - x1;
        int x = w / 2;
        int y = h / 2;
        
        draw_rectangle(image, W_INP, H_INP, x1, y1, x2, y2, 255);
    }

    /* ----------------------- SAVE IMAGE --------------------- */
    status = WriteImageToFile(
        STR(OUTPUT_FILE_NAME),
        W_INP, 
        H_INP, 
        CHANNELS, 
        image,
        RGB888_IO
    ); 
}