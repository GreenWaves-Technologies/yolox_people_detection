#include <pmsis.h>
#include "gaplib/jpeg_encoder.h"
#include "gaplib/ImgIO.h"
#include "main.h"

PI_L2 char jpeg_image[2048*50]; // how to set the size of the jpeg image? how to get this buffer the main.c file?

char * compress(uint8_t * image, int * size){

    jpeg_encoder_t enc;
    unsigned int image_size = W_INP * H_INP * CHANNELS;

        // Open JPEG encoder
    printf("Start JPEG encoding\n");

    struct jpeg_encoder_conf enc_conf;
    jpeg_encoder_conf_init(&enc_conf);

#ifdef RUN_ENCODER_ON_CLUSTER
    enc_conf.flags = JPEG_ENCODER_FLAGS_CLUSTER_OFFLOAD;
#else
    enc_conf.flags = 0x0;
#endif

    //For color Jpeg this flag can be added
    enc_conf.flags |= JPEG_ENCODER_FLAGS_COLOR;
    enc_conf.width  = W_INP;
    enc_conf.height = H_INP;

    if (jpeg_encoder_open(&enc, &enc_conf)){
        printf("Error opening JPEG encoder\n");
        // return -1;
    }

    if (jpeg_encoder_start(&enc)){
        printf("Error starting JPEG encoder\n");
        // return -1;
    }

    // Get the header so that we can produce full JPEG image
    pi_buffer_t bitstream;
    bitstream.data = &jpeg_image;
    bitstream.size = image_size;
    uint32_t header_size, footer_size, body_size;

    if (jpeg_encoder_header(&enc, &bitstream, &header_size)){
        printf("Error getting JPEG header\n");
        // return -1;
    }

    // // Now get the encoded image
    pi_buffer_t buffer;
    buffer.data    = image;
    buffer.size    = image_size;
    buffer.width   = W_INP;
    buffer.height  = H_INP;
    bitstream.data = &jpeg_image[header_size];

    pi_perf_conf(1<<PI_PERF_CYCLES);
    pi_perf_start();
    pi_perf_reset();

    if (jpeg_encoder_process(&enc, &buffer, &bitstream, &body_size)){
        printf("Error encoding JPEG image\n");
        // return -1;
    }

    pi_perf_stop();
    printf("Jpeg encoding done! Performance: %d Cycles\n", pi_perf_read(PI_PERF_CYCLES));    

    // An finally get the footer
    bitstream.data = &jpeg_image[body_size + header_size];
    if (jpeg_encoder_footer(&enc, &bitstream, &footer_size)){
        printf("Error getting JPEG footer\n");
        // return -1;
    }

    // calculate the total size and return it 
    int bitstream_size = body_size + header_size + footer_size;
    *size = bitstream_size;

    printf("Encoding done at addr %p size %d bytes\n", jpeg_image, bitstream_size);

    // close the endoer 
    jpeg_encoder_stop(&enc);
    jpeg_encoder_close(&enc);

    return jpeg_image;

}