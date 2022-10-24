#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "decoding.h"
#include "/home/abduragim/projects/gap_sdk/tools/autotiler_v3/DSP_Libraries/FastFloatApprox.h"

void decoding(f16 * Input, tTuple * feature_maps, f16 * strides, int size){

    int height, width, stride, i = 0;
    for (int j=0; j < size; j++){

        height = feature_maps[j].h;
        width = feature_maps[j].w;
        stride = strides[j];

        for(int h = 0; h < height; h++){
            for(int w = 0; w < width; w++){
                
                Input[i] = (Input[i] + w) * stride;
                Input[i + 1] = (Input[i + 1] + h) * stride;

                Input[i + 2] = fastexp(Input[i + 2]) * stride;
                Input[i + 3] = fastexp(Input[i + 3]) * stride;

                i += 6;
            }
        }
    }
}