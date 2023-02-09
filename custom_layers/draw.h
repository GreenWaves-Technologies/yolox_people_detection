// #include "Gap.h"

#define MAX(a, b)        (((a)>(b))?(a):(b))
#define MIN(a, b)        (((a)<(b))?(a):(b))

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

void draw_rectangle(
    unsigned char *Img, 
    int W, 
    int H, 
    int x, 
    int y, 
    int w, 
    int h, 
    int channels,
    unsigned char ColorValue);


void draw_boxes(
    float * model_L2_Memory_Dyn_casted,
    float * Output_1,
    int final_valid_boxes,
    int height,
    int width,
    int channels);

