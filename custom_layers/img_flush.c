#include "gaplib/fs_switch.h"
#include "img_flush.h"

int flush_iamge(char * image, int bitstream_size){

    struct pi_fs_conf host_fs_conf;
    pi_fs_conf_init(&host_fs_conf);
    struct pi_device host_fs;

    host_fs_conf.type = PI_FS_HOST;
    pi_open_from_conf(&host_fs, &host_fs_conf);

    if (pi_fs_mount(&host_fs))
        return -1;

    void *File = pi_fs_open(&host_fs, STR(OUTPUT_JPEG_FILE_NAME), PI_FS_FLAGS_WRITE);

    pi_fs_write(File, image, bitstream_size);        

    return 0;
}