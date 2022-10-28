#include "Gap.h"

typedef struct{
    f16 h;
    f16 w;
} tTuple;

void decoding(f16 * Input, tTuple * feature_maps, f16 * strides, int size);
