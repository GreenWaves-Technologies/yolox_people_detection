#include "draw.h"
#include "gaplib/ImgIO.h"

void draw_rectangle(
    unsigned char *Img, 
    int W, 
    int H, 
    int x, 
    int y, 
    int w, 
    int h, 
    int channels,
    unsigned char ColorValue
    ){

    int x0, x1, y0, y1;

    y0 = MAX(MIN(y, H - 1), 0);
    y1 = MAX(MIN(y + h - 1, H - 1), 0);

    x0 = x;
    y1 = h - 1;
    if (x0 >= 0 && x0 < W) {
        for (int i = y0; i <= y1; i++)
            for(int c = 0; c < channels; c++)
                Img[channels * (i * W + x0) + c] = ColorValue;
    }

    x1 = w - 1;
    if (x1 >= 0 && x1 < W) {
        for (int i = y0; i <= y1; i++)
            for(int c = 0; c < channels ; c++)
                Img[channels * (i * W + x1) + c] = ColorValue;
    }

    x0 = MAX(MIN(x, W - 1), 0);
    x1 = w - 1;

    y0 = y;
    if (y0 >= 0 && y0 < H) {
        for (int i = x0; i <= x1; i++)
            for(int c = 0; c < channels; c++)
                Img[channels * (y0 * W + i) + c] = ColorValue;
    }

    if (y1 >= 0 && y1 < H) {
        for (int i = x0; i <= x1; i++)
            for(int c = 0; c < channels; c++)
                Img[channels * (y1 * W + i) + c] = ColorValue;
    }
}

void draw_rectangle_new(
    unsigned char *Img, 
    int W, 
    int H, 
    int x, 
    int y, 
    int w, 
    int h, 
    int channels,
    unsigned char ColorValue
    )
{
    int line_width = 2;
    unsigned char color[3] = {0, 255, 255};
    //int color = 0xF800;
    /* top */
    for (int j=y; (j<H) && (j<(y+line_width)); j++) {
        for (int i=x; (i<W) && (i<(x+w)); i++) {
            Img[j*W*3 + i*3 + 0] = color[0];
            Img[j*W*3 + i*3 + 1] = color[1];
            Img[j*W*3 + i*3 + 2] = color[2];
        }
    }
    /* bottom */
    for (int j=y+h; (j<H) && (j<(y+h+line_width)); j++) {
        for (int i=x; (i<W) && (i<(x+w+line_width)); i++) {
            Img[j*W*3 + i*3 + 0] = color[0];
            Img[j*W*3 + i*3 + 1] = color[1];
            Img[j*W*3 + i*3 + 2] = color[2];
        }
    }
    /* left */
    for (int j=y; (j<H) && (j<(y+h)); j++) {
        for (int i=x; (i<W) && (i<(x+line_width)); i++) {
            Img[j*W*3 + i*3 + 0] = color[0];
            Img[j*W*3 + i*3 + 1] = color[1];
            Img[j*W*3 + i*3 + 2] = color[2];
        }
    }
    /* right */
    for (int j=y; (j<H) && (j<(y+h+line_width)); j++) {
        for (int i=x+w; (i<W) && (i<(x+w+line_width)); i++) {
            Img[j*W*3 + i*3 + 0] = color[0];
            Img[j*W*3 + i*3 + 1] = color[1];
            Img[j*W*3 + i*3 + 2] = color[2];
        }
    }
}

void draw_boxes_save(
    float * model_L2_Memory_Dyn_casted,
    float * Output_1,
    int final_valid_boxes,
    int height,
    int width, 
    int channels
    ){

    unsigned char * image = (unsigned char *) model_L2_Memory_Dyn_casted;
    int status = ReadImageFromFile(
        STR(INPUT_FILE_NAME),
        width, 
        height, 
        channels, 
        image,
        width * height * channels * sizeof(char), 
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

        draw_rectangle(image, width, height, x1, y1, x2, y2, channels, 255);
    }

    /* ----------------------- SAVE IMAGE --------------------- */
    status = WriteImageToFile(
        STR(OUTPUT_FILE_NAME),
        width, 
        height , 
        channels, 
        image,
        RGB888_IO // GRAY_SCALE_IO
    ); 
}

void draw_boxes(
    unsigned char * image,
    float * Output_1,
    int final_valid_boxes,
    int height,
    int width, 
    int channels
    ){

    for (int i=0; i < final_valid_boxes; i++){
        // get box
        int x1 = (int) Output_1[i*7 + 0];
        int y1 = (int) Output_1[i*7 + 1];
        int x2 = (int) Output_1[i*7 + 2];
        int y2 = (int) Output_1[i*7 + 3];

        // get score and class 
        float score =       Output_1[i*7 + 4] * Output_1[i*7 + 5];
        int   cls   = (int) Output_1[i*7 + 6];

        draw_rectangle_new(image, width, height, x1, y1, x2, y2, channels, 255);
    }
}