#include "Gap.h"

typedef struct{
    float h;
    float w;
} tTuple;

// void decoding(float * Input, tTuple * feature_maps, float * strides, int size);
void decoding(f16 * Input, tTuple * feature_maps, float * strides, int size);
