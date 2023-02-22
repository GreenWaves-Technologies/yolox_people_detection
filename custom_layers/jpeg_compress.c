#include "jpeg_compress.h"

void jpeg_init(jpeg_encoder_t *enc, int height, int width, pi_device_t cluster_dev, void *l1_memory){

    struct jpeg_encoder_conf enc_conf;
    jpeg_encoder_conf_init(&enc_conf);

    // enc_conf.flags = 0x0;

    //For color Jpeg this flag can be added
    enc_conf.flags |= JPEG_ENCODER_FLAGS_COLOR;
    enc_conf.width  = width;
    enc_conf.height = height;

    if (jpeg_encoder_open(enc, &enc_conf)){
        printf("Error opening JPEG encoder\n");
        // return -1;
    }

    // if (jpeg_encoder_start(enc)){
    //     printf("Error starting JPEG encoder\n");
    //     // return -1;
    // }
    // I don't need the start since I want to manually set cluster and l1 pointers:
    enc->cluster_dev = cluster_dev;
    enc->l1_constants = l1_memory;
    enc->cl_blocks = l1_memory + sizeof(JpegConstants);
    jpeg_copy_constants_to_l1(enc);
}

void jpeg_deinit(jpeg_encoder_t *enc){
    // close the endoer 
    // jpeg_encoder_stop(enc); Don't need to stop since L1 and Cluster are externally managed
    jpeg_encoder_close(enc);
}


int compress(jpeg_encoder_t *enc, uint8_t * image, uint8_t * jpeg_image, int * size, int height, int width, int channels){

    // Get the header so that we can produce full JPEG image
    unsigned int image_size = height * width * channels; 
    pi_buffer_t bitstream;
    bitstream.data = jpeg_image;
    bitstream.size = image_size;
    uint32_t header_size, footer_size, body_size;

    if (jpeg_encoder_header(enc, &bitstream, &header_size)){
        printf("Error getting JPEG header\n");
        return -1;
    }

    // // Now get the encoded image
    pi_buffer_t buffer;
    buffer.data    = image;
    buffer.size    = image_size;
    buffer.width   = width;
    buffer.height  = height;
    bitstream.data = &jpeg_image[header_size];

    if (jpeg_encoder_process(enc, &buffer, &bitstream, &body_size)){
        printf("Error encoding JPEG image\n");
        return -1;
    }

    // An finally get the footer
    bitstream.data = &jpeg_image[body_size + header_size];
    if (jpeg_encoder_footer(enc, &bitstream, &footer_size)){
        printf("Error getting JPEG footer\n");
        return -1;
    }

    // calculate the total size and return it 
    int bitstream_size = body_size + header_size + footer_size;
    *size = bitstream_size;
    return 0;
}