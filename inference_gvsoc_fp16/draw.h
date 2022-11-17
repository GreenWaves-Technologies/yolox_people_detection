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
    f16 * model_L2_Memory_Dyn_casted,
    f16 * Output_1,
    int final_valid_boxes);