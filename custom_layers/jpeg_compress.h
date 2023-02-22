#include <pmsis.h>
#include "gaplib/jpeg_encoder.h"
#include "gaplib/ImgIO.h"

void jpeg_init(jpeg_encoder_t *enc, int height, int width, pi_device_t cluster_dev, void *l1_memory);
void jpeg_deinit(jpeg_encoder_t *enc);
int compress(jpeg_encoder_t *enc, uint8_t * image, uint8_t * jpeg_image, int * size, int height, int width, int channels);