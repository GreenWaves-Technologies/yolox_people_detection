#include "Gap.h"

void draw_rectangle(
    unsigned char *Img, 
    int W, 
    int H, 
    int x, 
    int y, 
    int w, 
    int h, 
    unsigned char ColorValue);


void draw_boxes(
    float * model_L2_Memory_Dyn_casted,
    float * Output_1,
    int final_valid_boxes);