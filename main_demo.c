
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include "yolox.h"
#include "spi_comm.h"
#include "uart_comm.h"

#define STREAM_OVER_UART

// parameters needed for decoding layer
// !!! do not forget to change the stride sizes accordint to the input size !!!  
tTuple feature_maps[STRIDE_SIZE] = {{30.0, 40.0}, {15.0, 20.0}, {8.0, 10.0}};
float strides[STRIDE_SIZE] = {8.0, 16.0, 32.0};

// cycles count variables
unsigned int performances[8];

AT_DEFAULTFLASH_EXT_ADDR_TYPE yolox_L3_Flash = 0;

/* Outputs */
L2_MEM float Output_1[9480];
// L2_MEM unsigned char* Output_3;

//pi_device_t Ram;
AT_DEFAULTRAM_T DefaultRam;
static struct pi_default_ram_conf ram_conf;
uint32_t ext_ram_buf;
PI_L2 unsigned char *iter_buff[2];

extern L1_CL_MEM AT_L1_POINTER DeMosaic_L1_Memory;
extern L2_MEM AT_L2_POINTER DeMosaic_L2_Memory;

// 2 Rows
#define ITER_SIZE (W_CAM*2*BYTES_CAM)
#define RAW_SIZE (W_CAM*H_CAM*BYTES_CAM) // For now only 10 bits config works

pi_device_t* camera;
PI_L2 unsigned char *buff[2];

static void cluster()
{
    PRINTF("\t\t***Start CLUSTER ***\n");
    yoloxCNN(Output_1);
}

int main(int argc, char *argv[])
{
    PRINTF("\n\n\t *** NNTOOL main_demo ***\n\n");

    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = STACK_SIZE;

    cl_conf.id = 0; /* Set cluster ID. */
                    // Enable the special icache for the master core
    cl_conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |
                    // Enable the prefetch for all the cores, it's a 9bits mask (from bit 2 to bit 10), each bit correspond to 1 core
                    PI_CLUSTER_ICACHE_PREFETCH_ENABLE |
                    // Enable the icache for all the cores
                    PI_CLUSTER_ICACHE_ENABLE;

    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        PRINTF("Cluster open failed !\n");
        pmsis_exit(-4);
    }


    if(FREQ_FC >240 || FREQ_CL >240 || FREQ_PE >240)
        pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, 800);

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        PRINTF("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }
    if(FREQ_FC <=240 && FREQ_CL <=240 && FREQ_PE <=240)
        pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, 650);
    
    PRINTF("FC Frequency as %d Hz, CL Frequency = %d Hz, PERIIPH Frequency = %d Hz\n", 
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    PRINTF("Constructor\n");
    int ConstructorErr = yoloxCNN_Construct();
    if (ConstructorErr)
    {
        PRINTF("Graph constructor exited with error: %d\n(check the generated file mainKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }
    
    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

    // Output_3= pi_l2_malloc(320*240*3);
    // if(Output_3==NULL){
    //     printf("Output alloc error\n");
    //     return -1;
    // }

    #ifdef STREAM_OVER_UART
    pi_device_t uart_dev;
    init_uart_communication(&uart_dev,1152000);
    #else
    // Initialize SPI
    pi_device_t spi_slave;
    struct pi_spi_conf spi_slave_conf;
    spi_slave_init(&spi_slave, &spi_slave_conf);
    #endif


    //Open camera
    if (pi_open(PI_CAMERA_HM0360, &camera))
    {
        PRINTF("Failed to open camera\n");
        return -1;
    }
    PRINTF("Turning camera on...\n");
    //turn on camera
    pi_camera_control(camera, PI_CAMERA_CMD_ON, 0);
    // Allocate ping pong buffers for Camera read
    iter_buff[0] = pi_l2_malloc(ITER_SIZE);
    if (iter_buff[0] == NULL) return -1;
    iter_buff[1] = pi_l2_malloc(ITER_SIZE);
    if (iter_buff[1] == NULL) return -1;

    /* Init and opend jpeg encoder */
    jpeg_encoder_t enc;

    pi_gpio_flags_e flags = PI_GPIO_OUTPUT;
    //This is CN6 pin5
    pi_gpio_pin_configure(PI_PAD_048, flags);
    //This is GPIO
    //pi_gpio_pin_configure(PI_PAD_086, flags);
    
    pi_evt_t   end_of_capture_evt;
    int iter=0;
    char im_name[128];
    while(1){
        //pi_gpio_pin_toggle(PI_PAD_086);
        pi_gpio_pin_toggle(PI_PAD_048);

        pi_camera_capture_async(camera, (unsigned char*) Input_1, 640*480,pi_evt_sig_init(&end_of_capture_evt));
        pi_camera_control(camera, PI_CAMERA_CMD_START, 0);
        pi_evt_wait(&end_of_capture_evt);
        pi_camera_control(camera, PI_CAMERA_CMD_STOP, 0);

        //To save raw image decoment this:
        // sprintf(im_name,"../raw_%03d.pgm",iter);
        // WriteImageToFile(im_name, 640, 480, sizeof(uint8_t), Input_1, GRAY_SCALE_IO);
        
        pi_cluster_task(&task, (void (*)(void *))cluster, NULL);

        //To save Debayer image decomment this:        
        // sprintf(im_name,"../ISP_%03d.pgm",iter);
        // WriteImageToFile(im_name, 320, 240,3*sizeof(uint8_t), Output_3, RGB888_IO);

        PRINTF("\t\t***Start FC timer***\n");
        gap_fc_starttimer();
        gap_fc_resethwtimer();
        int perf_idx = 0;

        /* ------ INFERENCE ------*/
        PRINTF("\t\t***Call CLUSTER***\n");
        performances[perf_idx] = pi_time_get_us();
        pi_cluster_send_task_to_cl(&cluster_dev, &task);
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;

        /* ------ POST PROCESSING ------*/
        /* ---- DECODING (inplace) -----*/
        PRINTF("\t\t***Start decoding***\n");
        performances[perf_idx] = pi_time_get_us();
        decoding_and_xywh2xyxy(
            Output_1,
            feature_maps, 
            strides, 
            STRIDE_SIZE
        );
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;

        /* ------ filter boxes ------*/
        PRINTF("\t\t***Start filter boxes ***\n");
        //cast model_L2_Memory_Dyn to float16
        float * filtered_boxes = (float *) yolox_L2_Memory_Dyn;
        performances[perf_idx] = pi_time_get_us();
        int num_val_boxes = filter_boxes(
            Output_1,
            filtered_boxes, 
            CONF_THRESH, 
            RAWS
        );
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;

        /* ------ conver boxes ------*/
        PRINTF("\t\t***Start conver boxes ***\n");
        Box *bboxes = (Box *) (filtered_boxes + (RAWS * 6));
        int nms_valid_boxes = (num_val_boxes > top_k_boxes) ? (top_k_boxes) : (num_val_boxes);
        performances[perf_idx] = pi_time_get_us();
        to_bboxes(
            filtered_boxes,
            bboxes,
            nms_valid_boxes
        );
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;

        /* ------ nms ------*/
        PRINTF("\t\t***Start nms ***\n");
        performances[perf_idx] = pi_time_get_us();
        int final_valid_boxes = nms(
            bboxes, 
            Output_1,
            NMS_THRESH, 
            nms_valid_boxes
        );
        performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;

        //Draw rectangles and send trought UART
        draw_boxes((unsigned char *) Output_3, Output_1, final_valid_boxes, H_INP, W_INP, CHANNELS);

        /* ------ JPEG COMPRESSION ------ */
        
        PRINTF("\t\t***Start JPEG compression ***\n");
        int bitstream_size;
        
        uint8_t * jpeg_image = (uint8_t *)(yolox_L2_Memory_Dyn) + H_INP*W_INP*CHANNELS;
        if(final_valid_boxes){
            performances[perf_idx] = pi_time_get_us();
            jpeg_init(&enc, H_INP, W_INP, cluster_dev, yolox_L1_Memory);
            int jpeg_ret = compress(
                &enc,
                (uint8_t *) Output_3,
                jpeg_image,
                &bitstream_size,
                H_INP,
                W_INP,
                CHANNELS);
            performances[perf_idx] = pi_time_get_us() - performances[perf_idx]; perf_idx++;
        }else{
            performances[perf_idx] = 0; perf_idx++;
        }
        #ifdef STREAM_OVER_UART
        if(final_valid_boxes)
            send_jpeg_to_uart(&uart_dev, jpeg_image, bitstream_size, performances);
        else
            send_jpeg_to_uart(&uart_dev, jpeg_image, 0, performances);
        #else
        //send_jpeg_spi(&spi_slave,jpeg_image, bitstream_size,performances);
        #endif
        iter++;

    } //end of while 1
    pi_l2_free(iter_buff[0], ITER_SIZE);
    pi_l2_free(iter_buff[1], ITER_SIZE);
    jpeg_deinit(&enc);

    /* ------ END ------*/
    PRINTF("\t\t***Runner completed***\n");

    yoloxCNN_Destruct();

    PRINTF("Ended\n");
    pmsis_exit(0);
    return 0;
}
