typedef struct{
    float h;
    float w;
} tTuple;

void decoding(float * Input, tTuple * feature_maps, float * strides, int hw_strides_size);
