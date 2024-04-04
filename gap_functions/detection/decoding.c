#include "decoding.h"
#include "FastFloatApprox.h"

void decoding_and_xywh2xyxy(float * Input, tTuple * feature_maps, float * strides, int hw_strides_size){

    int height, width, stride, i = 0;
    for (int j=0; j < hw_strides_size; j++){

        height = feature_maps[j].h;
        width = feature_maps[j].w;
        stride = strides[j];

        for(int h = 0; h < height; h++){
            for(int w = 0; w < width; w++){
                float box_xcnt, box_ycnt, box_width, box_height;
                float x1, x2, y1, y2;
                
                box_xcnt   = (Input[i    ] + w) * stride;
                box_ycnt   = (Input[i + 1] + h) * stride;

                box_width  = fastexp(Input[i + 2]) * stride;
                box_height = fastexp(Input[i + 3]) * stride;
                // printf("[%d] %.2f %.2f %.2f %.2f %.2f %.2f\n", i, Input[i], Input[i+1], Input[i+2], Input[i+3], Input[i+4], Input[i+5]);

                x1 = box_xcnt - (box_width  / 2); 
                y1 = box_ycnt - (box_height / 2);
                x2 = box_xcnt + (box_width  / 2);
                y2 = box_ycnt + (box_height / 2);

                Input[i + 0] = x1;
                Input[i + 1] = y1;
                Input[i + 2] = x2;
                Input[i + 3] = y2;

                i += 6;
            }
        }
    }
}