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
    unsigned char ColorValueR,
    unsigned char ColorValueG,
    unsigned char ColorValueB
    ){

    int line_width = 2;
    int x0, x1, y0, y1;

    x0 = MAX(MIN(x, W - 1), 0);
    x1 = MAX(MIN(w - 1, W - 1), 0);
    y0 = MAX(MIN(y, H - 1), 0);
    y1 = MAX(MIN(h - 1, H - 1), 0);

    // left
    for (int i = y0; i < y1; i++) {
        for (int j = x0; j<(x0+line_width); j++) {
            Img[3 * (i * W + j) + 0] = ColorValueR;
            Img[3 * (i * W + j) + 1] = ColorValueG;
            Img[3 * (i * W + j) + 2] = ColorValueB;
        }
    }

    // right
    for (int i = y0; i < y1; i++) {
        for (int j = x1-line_width; j<x1; j++) {
            Img[3 * (i * W + j) + 0] = ColorValueR;
            Img[3 * (i * W + j) + 1] = ColorValueG;
            Img[3 * (i * W + j) + 2] = ColorValueB;
            // for(int c = 0; c < channels ; c++)
            //     Img[channels * (i * W + x1) + c] = ColorValueR;
        }
    }

    // top
    for (int i = y0; i < y0+line_width; i++) {
        for (int j = x0; j < x1; j++) {
            Img[3 * (i * W + j) + 0] = ColorValueR;
            Img[3 * (i * W + j) + 1] = ColorValueG;
            Img[3 * (i * W + j) + 2] = ColorValueB;
            // for(int c = 0; c < channels; c++)
            //     Img[channels * (y0 * W + i) + c] = ColorValueR;
        }
    }

    // bottom
    for (int i = y1-line_width; i < y1; i++) {
        for (int j = x0; j < x1; j++) {
            Img[3 * (i * W + j) + 0] = ColorValueR;
            Img[3 * (i * W + j) + 1] = ColorValueG;
            Img[3 * (i * W + j) + 2] = ColorValueB;
            // for(int c = 0; c < channels; c++)
            //     Img[channels * (y1 * W + i) + c] = ColorValueR;
        }
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

        draw_rectangle(image, width, height, x1, y1, x2, y2, channels, 255, 255, 0);
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

        draw_rectangle(image, width, height, x1, y1, x2, y2, channels, 255, 255, 0);
    }
}